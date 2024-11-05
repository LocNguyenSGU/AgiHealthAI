from kivy.app import App
from kivy.uix.label import Label

class HelloWorldApp(App):
    def build(self):
        # Tạo một label với văn bản "Hello, World!"
        return Label(text="Hello, World!")

# Khởi chạy ứng dụng
if __name__ == '__main__':
    HelloWorldApp().run()