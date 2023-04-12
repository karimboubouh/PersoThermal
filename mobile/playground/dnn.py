from time import sleep

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.lang import Builder
import threading

Builder.load_string('''

<MyBox>:
    orientation: 'horizontal'
    cols: 2
    Label:
        text: root.tobeupd
    Button:
        text: 'Start Update'
        on_release: root.upd_ltxt()

''')


class MyBox(BoxLayout):
    # tobeupd = StringProperty()

    def __init__(self):
        super(MyBox, self).__init__()
        self.tobeupd = '#'
        threading.Thread(target=self.update_label).start()

    def upd_ltxt(self):
        pass

    def update_label(self):
        for i in range(1, 10):
            print(self.tobeupd)
            self.tobeupd = str(i)
            sleep(1)


class updApp(App):
    def build(self):
        return MyBox()


if __name__ == '__main__':
    updApp().run()
