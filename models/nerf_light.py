import pytorch_lightning as pl
import torch.optim as optim
import torch
from pytorch3d.renderer import (
    RayBundle,
    ray_bundle_to_ray_points,
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher
)
from utils.helpers import (
    huber,
    sample_images_at_mc_locs
)
from pytorch3d.renderer import ImplicitRenderer
from PIL import Image as im

class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
    
class Nerf(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.n_harmonic_functions = config.model.n_harmonic_functions
        self.n_hidden_neurons = config.model.n_harmonic_functions
        self.embedding_dim = self.n_harmonic_functions * 2 * 3
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.harmonic_embedding = HarmonicEmbedding(self.n_harmonic_functions)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
        )

        self.density_layer[0].bias.data[0] = -1.5

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
        # rays_points_world.shape = [minibatch x ... x 3]
        
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        
        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        
        # Finally, given the per-point features, 
        # execute the density and color branches.
        
        rays_densities = self._get_densities(features)
        rays_colors = torch.ones(rays_densities.shape[0], rays_densities.shape[1], rays_densities.shape[2], 3).to(self.device)
        
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
        silhouettes, K, R, t = train_batch
        silhouettes = torch.movedim(silhouettes, 1, -1)
        batch_cameras = FoVPerspectiveCameras(K=K, R=R, T=t, device=self.device)

        # Evaluate the nerf model.
        rendered_silhouettes_, sampled_rays = self.renderer_mc(
            cameras=batch_cameras, 
            volumetric_function=self.forward
        )
        
        _, rendered_silhouettes = (rendered_silhouettes_.split([3,1], dim=-1))
        
        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            silhouettes, 
            sampled_rays.xys
        )
        
        sil_err = huber(
        rendered_silhouettes, 
        silhouettes_at_rays,
        ).abs().mean()

        consistency_loss = huber(
            rendered_silhouettes.sum(axis=0), 
            silhouettes_at_rays.sum(axis=0),
        ).abs().mean()
        
        # The optimization loss is a simple sum of the color and silhouette errors.
        loss = sil_err

        # ------------ LOGGING -----------
        # Train Loss
        self.log('train_loss', loss, on_step=True, batch_size=self.batch_size)

        with torch.no_grad():
            if self.global_step % 200:
                eval_K = K[None, 0, ...]
                eval_R = R[None, 0, ...]
                eval_t = t[None, 0, ...]
                full_silhouette, _ =  self.renderer_grid(
                cameras=FoVPerspectiveCameras(K=eval_K, R=eval_R, T=eval_t, device=self.device),
                volumetric_function=self.batched_forward
                )
                _, full_silhouette = (full_silhouette.split([3,1], dim=-1))
                
                clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
                silhouette_image = clamp_and_detach(full_silhouette[...,0])
                # tensorboard = self.logger.experiment
                # tensorboard.add_image(silhouette_image)
                self.logger.experiment.add_image('silhouette image', silhouette_image)

        return loss

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()
    
    