from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivymd.app import MDApp
from src.screens import *

# Window.size = (336, 600)
# Window.release_all_keyboards()


class ThermalApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.primary_hue = "700"
        return Builder.load_file('src/template.kv')


ThermalApp().run()
