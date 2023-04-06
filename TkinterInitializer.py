import tkinter as tk
import cv2
from PIL import Image, ImageTk

class TkinterInitializer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("914x636+129+91")
        self.root.minsize(120, 1)
        self.root.maxsize(1370, 749)
        self.root.resizable(1,  1)
        self.root.title("Face Recognition")

        self.Frame1 = tk.Frame(self.root)
        self.Frame1.place(relx=0.0, rely=0.0, relheight=0.983, relwidth=0.224)
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#d9d9d9")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")
        self.Frame2 = tk.Frame(self.root)
        self.Frame2.place(relx=0.219, rely=0.0, relheight=0.511, relwidth=0.509)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#d9d9d9")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")
        self.Label3 = tk.Label(self.Frame2)
        self.Label3.place(relx=0.0, rely=0.0, height=319, width=461)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(anchor='w')
        self.Label3.configure(background="#d9d9d9")
        self.Label3.configure(compound='left')
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(text='''Label''')
        self.Frame3 = tk.Frame(self.root)
        self.Frame3.place(relx=0.722, rely=0.0, relheight=0.998, relwidth=0.279)
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")
        self.Frame3.configure(highlightbackground="#d9d9d9")
        self.Frame3.configure(highlightcolor="black")
        self.Label1 = tk.Label(self.Frame3)
        self.Label1.place(relx=0.039, rely=0.0, height=222, width=244)
        self.Label1.configure(activebackground="#400040")
        self.Label1.configure(anchor='w')
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(compound='left')
        self.Label1.configure(disabledforeground="#400040")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Label''')
        self.Label2 = tk.Label(self.Frame3)
        self.Label2.place(relx=0.275, rely=0.378, height=21, width=174)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(anchor='w')
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(compound='left')
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 11 -slant italic")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Label''')
        self.Label4 = tk.Label(self.Frame3)
        self.Label4.place(relx=0.039, rely=0.425, height=21, width=84)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(anchor='w')
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(compound='left')
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(font="-family {Segoe UI} -size 10 -weight bold")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(highlightbackground="#d9d9d9")
        self.Label4.configure(highlightcolor="black")
        self.Label4.configure(text='''Arrival time :''')
        self.Label5 = tk.Label(self.Frame3)
        self.Label5.place(relx=0.039, rely=0.378, height=21, width=54)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(anchor='w')
        self.Label5.configure(background="#d9d9d9")
        self.Label5.configure(compound='left')
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(font="-family {Segoe UI} -size 10 -weight bold")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(highlightbackground="#d9d9d9")
        self.Label5.configure(highlightcolor="black")
        self.Label5.configure(text='''Name :''')
        self.Label6 = tk.Label(self.Frame3)
        self.Label6.place(relx=0.392, rely=0.425, height=21, width=144)
        self.Label6.configure(anchor='w')
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(compound='left')
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(font="-family {Segoe UI} -size 11 -slant italic")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(text='''Label6''')
        self.Listbox1 = tk.Listbox(self.root)
        self.Listbox1.place(relx=0.208, rely=0.503, relheight=0.491, relwidth=0.519)
        self.Listbox1.configure(background="#d9d9d9")
        self.Listbox1.configure(disabledforeground="#a3a3a3")
        self.Listbox1.configure(font="TkFixedFont")
        self.Listbox1.configure(foreground="#000000")
        self.Listbox1.configure(relief="flat")

    def update_listbox1(self,index, text):
        self.Listbox1.insert(index,text)
      
    def update_label_image(self, image):
        # Convert the OpenCV frame to a PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Update the label widget with the new image
     
        photo = ImageTk.PhotoImage(pil_image)
        self.Label3.config(image=photo)
        self.Label3.image = photo


    def update_name_label(self, text):
        self.Label2.configure(text=text)
    def update_time_label(self, text):
        self.Label6.configure(text=text)
    def update_current_user_label(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Update the label widget with the new image
     
        photo = ImageTk.PhotoImage(pil_image)
        self.Label1.config(image=photo)
        self.Label1.image = photo
