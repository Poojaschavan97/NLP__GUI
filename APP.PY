from tkinter import *
from tkinter.messagebox import showerror

# Importing custom database and API modules
from myDB import Database
from tkinter import messagebox
from myapi import API

class NLPAPP:

    def __init__(self):

        # Create a Database object
        self.dbo=Database()  # This can now access any method from the Database class

        # Fetch API object
        self.apio=API()

        # Load login GUI
        self.root=Tk()  # Create the main window
        self.root.title('NLPAPP')  # Title in the window header
        self.root.iconbitmap('RESOURCES/favicon.ico')  # Adding favicon
        self.root.geometry('500x500')  # Set window size
        self.root.configure(bg='#c1dfd1')  # Set background color
        self.login_gui()  # Initialize login GUI
        self.root.mainloop()  # Run the GUI event loop

    # Define the Login GUI layout
    def login_gui(self):
        self.clear()  # Clear any existing GUI elements

        # Add a heading
        heading=Label(self.root, text='NLPApp')
        heading.pack(pady=(30,20))
        heading.configure(font=('Castellar',24,'bold'), bg='#c1dfd1',fg='purple')

        # Email input
        label1=Label(self.root,text='Enter Email')
        label1.pack(pady=(10,10))
        self.email_input=Entry(self.root,width=50)
        self.email_input.pack(pady=(5,10),ipady=3)

        # Password input
        label2 = Label(self.root, text='Enter Password')
        label2.pack(pady=(10, 10))
        self.password_input = Entry(self.root, width=50,show='*')
        self.password_input.pack(pady=(5, 10), ipady=3)

        # Login button
        login_button=Button(self.root,text='Login',width=10,height=1, command=self.perform_login)
        login_button.pack(pady=(10,10))

        # Registration redirection
        label3=Label(self.root,text='Not a member?')
        label3.pack(pady=(20,10))
        redirect_button = Button(self.root, text='Register Now', width=10, height=1,command=self.register_gui)
        redirect_button.pack(pady=(10, 10))

    # Define the Registration GUI layout
    def register_gui(self):
        self.clear()

        # Add a heading
        heading = Label(self.root, text='NLPApp')
        heading.pack(pady=(30, 20))
        heading.configure(font=('Castellar', 24, 'bold'), bg='#c1dfd1',fg='purple')

        # Name input
        label0= Label(self.root, text='Enter Name')
        label0.pack(pady=(10, 10))
        self.name_input = Entry(self.root, width=50)
        self.name_input.pack(pady=(5, 10), ipady=3)

        # Email input
        label1 = Label(self.root, text='Enter Email')
        label1.pack(pady=(10, 10))
        self.email_input = Entry(self.root, width=50)
        self.email_input.pack(pady=(5, 10), ipady=3)

        # Password input
        label2 = Label(self.root, text='Enter Password')
        label2.pack(pady=(10, 10))
        self.password_input = Entry(self.root, width=50, show='*')
        self.password_input.pack(pady=(5, 10), ipady=3)

        # Register button
        register_button = Button(self.root, text='Register', width=10, height=1, command=self.perform_registration)
        register_button.pack(pady=(10, 10))

        # Login redirection
        label3 = Label(self.root, text='Already a member?')
        label3.pack(pady=(20, 10))
        redirect_button = Button(self.root, text='Login Now', width=10, height=1, command=self.login_gui)
        redirect_button.pack(pady=(10, 10))

    # Function to clear the current GUI
    def clear(self):
        for i in self.root.pack_slaves():
            i.destroy()

    # Function to handle registration logic
    def perform_registration(self):
        name=self.name_input.get()
        email=self.email_input.get()
        password=self.password_input.get()

        response=self.dbo.add_data(name,email,password)

        if response:
            messagebox.showinfo('Success', 'Registration successful. You can log in now.')
        else:
            messagebox.showerror('Error', 'Email ID already exists.')

    # Function to handle login logic
    def perform_login(self):
        email=self.email_input.get()
        password=self.password_input.get()

        response=self.dbo.search(email,password)
        if response:
            messagebox.showinfo('Success', 'Login successful.')
            self.home_gui()
        else:
            messagebox,showerror('Error', 'Incorrect email/password.')

    # Define the Home GUI layout
    def home_gui(self):
        self.clear()

        heading = Label(self.root, text='NLPApp')
        heading.pack(pady=(30, 20))
        heading.configure(font=('Castellar', 24, 'bold'), bg='#c1dfd1',fg='purple')

        # Options for sentiment analysis, NER, and summary creation
        sentiment_btn=Button(self.root, text='Sentiment Analysis',width=50,height=4, command=self.sentiment_gui)
        sentiment_btn.pack(pady=(10,10))
        ner_btn = Button(self.root, text='Named Entity Recognition', width=50, height=4, command=self.ner_gui)
        ner_btn.pack(pady=(10, 10))
        summary_btn = Button(self.root, text='Summary Creator', width=50, height=4, command=self.summary_gui)
        summary_btn.pack(pady=(10, 10))
        logout_btn = Button(self.root, text='Logout', width=30, height=2,command=self.login_gui)
        logout_btn.pack(pady=(10, 10))

    # Define GUI for sentiment analysis
    def sentiment_gui(self):
        self.clear()

        heading = Label(self.root, text='NLPApp')
        heading.pack(pady=(20, 10))
        heading.configure(font=('Castellar', 24, 'bold'), bg='#c1dfd1',fg='purple')

        heading1 = Label(self.root, text='Sentiment Analysis')
        heading1.pack(pady=(10, 10))
        heading1.configure(font=('Forte', 18, 'bold'), bg='#c1dfd1',fg='coral')

        # Text input for sentiment analysis
        label1 = Label(self.root, text='Enter Text')
        label1.pack(pady=(20, 10))
        self.sentiment_input = Entry(self.root, width=50)
        self.sentiment_input.pack(pady=(5, 10), ipady=3)

        # Analyze button
        sentiment_btn = Button(self.root, text='Analyze Sentiment', width=20, height=1, command=self.do_sentiment_analysis)
        sentiment_btn.pack(pady=(10, 10))

        self.response= Label(self.root, text='',bg='#c1dfd1')
        self.response.pack(pady=(20, 10))
        self.response.configure(font=('Arial Narrow',16))

        # Back button
        back_btn = Button(self.root, text='Back', width=10, height=1, command=self.home_gui)
        back_btn.pack(pady=(10, 10))

    # Function to perform sentiment analysis
    def do_sentiment_analysis(self):
        text=self.sentiment_input.get()
        result=self.apio.sentiment_analysis(text)

        txt=''
        for i in result['scored_labels']:
            txt=txt+i['label']+' --> '+str(i['score'])+'\n'

        self.response['text']=txt

    # GUI for named entity recognition
    def ner_gui(self):
        self.clear()

        # Rest of the function follows a similar structure as sentiment analysis

    # GUI for summary creation
    def summary_gui(self):
        self.clear()

        # Rest of the function follows a similar structure as sentiment analysis

# Initialize the application
nlp=NLPAPP()
