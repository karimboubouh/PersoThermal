from kivy.uix.screenmanager import Screen

from src.utils import fixed_seed


class WelcomeScreen(Screen):

    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)
        fixed_seed(True)
