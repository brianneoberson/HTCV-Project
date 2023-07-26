import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders.silhouette_dataloader import SilhouetteDataset
from dataloaders.imgsilpair_dataloader import ImgSilPairDataset
from pytorch3d.renderer import (
    RayBundle,
    ray_bundle_to_ray_points,
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    AbsorptionOnlyRaymarcher,
    ImplicitRenderer
)
from utils.helpers import (
    huber,
    sample_images_at_mc_locs
)
import numpy as np
from harmonic_embedding import HarmonicEmbedding

class NerfColor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.model.n_harmonic_functions * 2 * 3
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.harmonic_embedding = HarmonicEmbedding(config.model.n_harmonic_functions)
        layers = []
        layers.append(torch.nn.Linear(self.embedding_dim, config.model.n_hidden_neurons))
        layers.append(torch.nn.Softplus(beta=10.0))
        for l in range(self.config.model.n_hidden_layers): 
            layers.append(torch.nn.Linear(config.model.n_hidden_neurons, config.model.n_hidden_neurons))
            layers.append(torch.nn.Softplus(beta=10.0))

        self.mlp = torch.nn.Sequential(*layers)

        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(config.model.n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
        )

        self.density_layer[0].bias.data[0] = -1.5
        
        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(config.model.n_hidden_neurons + self.embedding_dim, config.model.n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(config.model.n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )

        raysampler_grid = NDCMultinomialRaysampler(
            image_height=config.model.render_size,
            image_width=config.model.render_size,
            n_pts_per_ray=config.model.nb_samples_per_ray,
            min_depth=config.model.min_depth,
            max_depth=config.model.volume_extent_world,
        )
        raysampler_mc = MonteCarloRaysampler(
            min_x = -1.0,
            max_x = 1.0,
            min_y = -1.0,
            max_y = 1.0,
            n_rays_per_image=config.model.nb_rays_per_image,
            n_pts_per_ray=config.model.nb_samples_per_ray,
            min_depth=config.model.min_depth,
            max_depth=config.model.volume_extent_world,
        )

        raymarcher = EmissionAbsorptionRaymarcher()
        #raymarcher = AbsorptionOnlyRaymarcher()
        self.renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
        self.renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

    def forward(
            self, 
            ray_bundle: RayBundle, 
            **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's 
        RGB color and opacity respectively.
        
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        embeds = self.harmonic_embedding(
            rays_points_world
        )

        features = self.mlp(embeds)
        
        rays_densities = self._get_densities(features)
        #rays_colors = torch.ones(rays_densities.shape[0], rays_densities.shape[1], rays_densities.shape[2], 3).to(self.device)
        rays_colors = self._get_colors(features, ray_bundle.directions)
        
        return rays_densities, rays_colors 
        
    def batched_forward(
        self, 
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,        
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.
        
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), self.batch_size)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]
        
        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0,1)
        ]
        
        return rays_densities, rays_colors
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, silhouettes, K, R, t = train_batch
        silhouettes = torch.movedim(silhouettes, 1, -1)
        batch_cameras = FoVPerspectiveCameras(K=K, R=R, T=t, device=self.device)


        rendered_images_silhouettes, sampled_rays = self.renderer_mc(
            cameras=batch_cameras, 
            volumetric_function=self.forward
        )
        
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )
        
        silhouettes_at_rays = sample_images_at_mc_locs(
            silhouettes, 
            sampled_rays.xys
        )
        
        sil_err = huber(
            rendered_silhouettes, 
            silhouettes_at_rays,
        ).abs().mean()
        
        colors_at_rays = sample_images_at_mc_locs(
            images, # otherwise has dim = 5
            sampled_rays.xys
        )
 
        color_err = huber(
            rendered_images, 
            colors_at_rays,
        ).abs().mean()

        consistency_err = huber(
            rendered_silhouettes.sum(axis=0), 
            silhouettes_at_rays.sum(axis=0),
        ).abs().mean()

       #Computing the custom Loss
        num_zeros = torch.sum(silhouettes_at_rays <= 0.5)
        num_ones = torch.sum(silhouettes_at_rays > 0.5)

        custom_err = (len(batch_cameras) * num_zeros * sil_err + num_ones * sil_err).abs().mean()
        
        loss = \
            self.config.trainer.lambda_sil_err * sil_err \
            + self.config.trainer.lambda_color_err * color_err

        # ------------ LOGGING -----------
        self.log('losses/train_loss', loss, on_step=True, batch_size=self.batch_size)
        self.log('losses/sil_err', sil_err, on_step=True, batch_size=self.batch_size)
        self.log('losses/color_err', color_err, on_step=True, batch_size=self.batch_size)


        with torch.no_grad():
            if self.current_epoch % self.config.trainer.log_image_every_n_epochs == 0:
                eval_K = K[None, 0, ...]
                eval_R = R[None, 0, ...]
                eval_t = t[None, 0, ...]
                color_image, target_image, silhouette_image, target_silhouette = self._render_image(
                    camera=FoVPerspectiveCameras(K=eval_K, R=eval_R, T=eval_t, device=self.device),
                    silhouette=silhouettes[0].unsqueeze(0),
                    image=images[0].unsqueeze(0)
                    )
                self.logger.experiment.add_image(
                    'Color Prediction vs Ground Truth', 
                    np.concatenate((color_image, target_image), axis=1), 
                    global_step=self.current_epoch, 
                    dataformats='HWC' )
                self.logger.experiment.add_image(
                    'Silhouette Prediction vs Ground Truth', 
                    np.concatenate((silhouette_image, target_silhouette), axis=1), 
                    global_step=self.current_epoch, 
                    dataformats='HWC')
                self.logger.experiment.add_histogram("Silhouette Values Histogram", silhouette_image, global_step=self.current_epoch, bins='auto')
        return loss

    def _render_image(self, camera, silhouette, image):
        rendered_image_silhouette, sampled_rays =  self.renderer_grid(
                cameras=camera,
                volumetric_function=self.batched_forward
                )
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )
        silhouettes_at_rays = sample_images_at_mc_locs(
            silhouette, 
            sampled_rays.xys
        )
        colors_at_rays = sample_images_at_mc_locs(
            image, 
            sampled_rays.xys
        )
        clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
        silhouette_image = clamp_and_detach(rendered_silhouette)
        color_image = clamp_and_detach(rendered_image)
        target_silhouette = clamp_and_detach(silhouettes_at_rays[0])
        target_image = clamp_and_detach(colors_at_rays[0])
        return color_image, target_image, silhouette_image, target_silhouette

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()
    
    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        
        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions 
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]
        
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )
        
        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )
        
        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )
        
        # Concatenate ray direction embeddings with 
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Get the total number of iterations
        total_iterations = len(self.train_dataloader()) * self.config.trainer.max_epochs

        # Compute the iteration at which the adjustment should be made
        iteration_threshold = round(total_iterations * 0.75)

        # Check if the current iteration matches the condition
        if self.global_step == iteration_threshold:
            print('Decreasing LR 10-fold ...')

            # Access the optimizer
            optimizer = self.optimizers()

            # Update the learning rate
            new_lr = self.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def train_dataloader(self):
        #dataset = SilhouetteDataset(self.config)
        dataset = ImgSilPairDataset(self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.trainer.batch_size, shuffle=True)
        return dataloader
    