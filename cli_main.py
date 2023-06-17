from pytorch_lightning.cli import LightningCLI

from base import EncoderModule

#from dataset.datamodule import ExpressionDatamodule
#from utils.save_config_callback import MySaveConfigCallback


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.test_set", "model.test_set")


def cli_main():
    cli = LightningCLI(model_class=EncoderModule)

if __name__ == '__main__':
    cli_main()