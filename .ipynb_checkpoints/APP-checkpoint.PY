from tkinter import *

class NLPAPP:

    def __init__(self):
        #load login GUI
        self.root=Tk()  # STORING OBJECTS OF TK
        self.root.title('NLPAPP')
        self.root.iconbitmap('RESOURCES/favicon.ico')
        self.root.geometry('500x500')
        self.root.configure(bg='#c1dfd1')
        self.login_gui()
        self.root.mainloop()  # to see GUI

    def login_gui(self):
        pass


nlp=NLPAPP()

