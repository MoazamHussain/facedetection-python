from tkinter import *
import os
from tkinter import filedialog
import cv2
from matplotlib import pyplot as plt


def main_screen():
    global screen
    screen = Tk()
    screen.title("My Application")
    screen.geometry("1350x728")
    filename = PhotoImage(file="Capture2.gif")
    background_label=Label(screen, image=filename)
    background_label.place(x=0 ,y=0,relwidth=1,relheight=1)
    Label(screen, text="Sign in\n to continue to Application",bg="indigo" ,fg="black", font=("Arial Bold",16)).pack()
    Button(screen, text="Login", bg="white", fg="indigo", font=("Arial Bold",12),width=8,height=2, command=login).place(x=650,y=370)
    Button(screen, text="Create account", bg="white", fg="indigo", font=("Arial Bold",12),width=14,height=2, command=register).place(x=620,y=425)

    screen.mainloop()


def delete2():
    screen3.destroy()


def delete3():
    screen4.destroy()


def delete4():
    screen5.destroy()


def delete5():
    screen6.destroy()


def delete6():
    screen7.destroy()


def logout():
    screen8.destroy()
    screen2.destroy()



def session():
    global screen8
    screen8 = Toplevel(screen)
    screen8.title("Page")
    screen8.geometry("1350x728")
    Label(screen8, text="Welcome to page",bg="steelblue",fg="black", width=300, height=5, font=("Arial Bold",16)).pack(side=TOP)
    Label(screen8, text="",bg="steelblue",fg="black", width=5, height=400, font=("Arial Bold",16)).pack(side=RIGHT)
    Label(screen8, text="", bg="steelblue", fg="black", width=5, height=400, font=("Arial Bold", 16)).pack(side=LEFT)
    Label(screen8, text="").pack()
    Label(screen8, text="").pack()
    Button(screen8, text="Begin",bg="white",fg="blue", width=8, height=2,command=Unknown_person).pack()
    Label(screen8, text="").pack()
    Button(screen8, text="Logout", fg="blue", width=10, height=1,command=logout).pack()
    Label(screen8, text="Partial Face Detection",bg="steelblue",fg="black", width=250, height=5, font=("Arial Bold",16)).pack(side=BOTTOM)


def Unknown_person():
    import face_recognition
    import cv2
    import numpy as np

    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    narendramodi_image = face_recognition.load_image_file("image/m1.jpg")
    narendramodi_face_encoding = face_recognition.face_encodings(narendramodi_image)[0]

    # Load a second sample picture and learn how to recognize it.
    kalamsir_image = face_recognition.load_image_file("image/s1.jpg")
    kalamsir_face_encoding = face_recognition.face_encodings(kalamsir_image)[0]

    p1 = face_recognition.load_image_file("image/a1.jpg")
    p1_face_encoding = face_recognition.face_encodings(p1)[0]

    p2 = face_recognition.load_image_file("image/p1.jpg")
    p2_face_encoding = face_recognition.face_encodings(p2)[0]

    p3 = face_recognition.load_image_file("image/k1.jpg")
    p3_face_encoding = face_recognition.face_encodings(p3)[0]

    p4 = face_recognition.load_image_file("image/S2.jpg")
    p4_face_encoding = face_recognition.face_encodings(p4)[0]



    # Create arrays of known face encodings and their names
    known_face_encodings = [
        narendramodi_face_encoding,
        kalamsir_face_encoding,
        p1_face_encoding,
        p2_face_encoding,
        p3_face_encoding,
        p4_face_encoding

    ]

    known_face_names = [
        "misbah",
        "sharavani",
        "aishwarya",
        "pratiksha",
        "kajal",
        "simran"

    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    nm = ""

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]



                face_names.append(name)


        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if (name == "Unknown"):
                nm = "Intruder!!!!!"
            else:
                nm = "Hello!" + name


        # Display the resulting image
        cv2.imshow('Video', frame)



        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    global screennew
    screennew = Toplevel(screen)
    screennew.title("Face Recognition")
    screennew.geometry("1350x728")
    Label(screennew, text=nm, fg="black", font="Arial").pack()


def login_successfully():
    session()


def username_not_found():
    global screen3
    screen3 = Toplevel(screen)
    screen3.title("Email-ID not found")
    screen3.geometry("250x100")
    Label(screen3, text="Please select register username", fg="black", font="Arial").pack()
    Label(screen3, text="").pack()
    Label(screen3, text="").pack()
    Button(screen3, text="ok", width=5, height=1, command=delete2).pack()


def Please_enter_correct_password():
    global screen4
    screen4 = Toplevel(screen)
    screen4.title("Password Error")
    screen4.geometry("250x100")
    Label(screen4, text="Please enter correct password", fg="black", font="Arial").pack()
    Label(screen4, text="").pack()
    Label(screen4, text="").pack()
    Button(screen4, text="ok", width=5, height=1, command=delete3).pack()


def passward_accepted():
    global screen5
    screen5 = Toplevel(screen)
    screen5.title("Success")
    screen5.geometry("250x100")
    Label(screen5, text="Registration successful", fg="black", font="Arial").pack()
    Label(screen5, text="").pack()
    Label(screen5, text="").pack()
    Button(screen5, text="ok", width=5, height=1, command=delete4).pack()


def Please_select_different_password():
    global screen6
    screen6 = Toplevel(screen)
    screen6.title("Success")
    screen6.geometry("250x100")
    Label(screen6, text="Please select different password", fg="black", font="Arial").pack()
    Label(screen6, text="").pack()
    Label(screen6, text="").pack()
    Button(screen6, text="ok", width=5, height=1, command=delete5).pack()


def Username_is_not_available():
    global screen7
    screen7 = Toplevel(screen)
    screen7.title("Success")
    screen7.geometry("1350x728")
    Label(screen7, text="Username is already taken", fg="black", font="Arial").pack()
    Label(screen7, text="").pack()
    Label(screen7, text="").pack()
    Button(screen7, text="ok", width=5, height=1, command=delete6).pack()


def register_user():
    global email_info
    global username_info
    global password_info
    email_info = email.get()
    username_info = username.get()
    password_info = password.get()
    list_of_files = os.listdir()
    if username_info not in list_of_files:
        file = open(username_info, "w")
        file.write(username_info + "\n")
        if email_info not in list_of_files:
            file = open(username_info, "a")
            file.write(email_info + "\n")
            if password_info not in list_of_files:
                file = open(username_info, "a")
                file.write(password_info + "\n")
                passward_accepted()
            else:
                Please_select_different_password()
        else:
            Email_id_already_register()
    else:
        Username_is_not_available()
    username_entry.delete(0, END)
    password_entry.delete(0, END)
    email_entry.delete(0, END)
    Label(screen1, text="Registration Success", fg="black").pack()


def Email_id_already_register():
    global screen13
    screen13 = Toplevel(screen)
    screen13.title("Success")
    screen13.geometry("1350x728")
    Label(screen13, text="Please select different password", fg="black", font="Arial").pack()
    Label(screen13, text="").pack()
    Label(screen13, text="").pack()
    Button(screen13, text="ok", width=5, height=1).pack()


def register():
    global screen1
    screen1 = Toplevel(screen)
    screen1.title("Register")
    screen1.geometry("1350x728")
    global username
    global password
    global email
    global username_entry
    global password_entry
    global email_entry
    username = StringVar()
    password = StringVar()
    email = StringVar()
    Label(screen1, text="Please Enter Details", bg="steelblue", fg="black", width=300, height=5,font=("Arial Bold", 16)).pack(side=TOP)
    Label(screen1, text="", bg="steelblue", fg="black", width=5, height=100, font=("Arial Bold", 16)).pack(side=RIGHT)
    Label(screen1, text="", bg="steelblue", fg="black", width=5, height=100, font=("Arial Bold", 16)).pack(side=LEFT)
    Label(screen1, text="", font=("Times New Roman",)).pack()
    Label(screen1, text="Email *", font=("Times New Roman",)).pack()
    email_entry = Entry(screen1, textvariable=email)
    email_entry.pack()
    Label(screen1, text="Username *", font=("Times New Roman",)).pack()
    username_entry = Entry(screen1, textvariable=username)
    username_entry.pack()
    Label(screen1, text="Password *", font=("Times New Roman",)).pack()
    password_entry = Entry(screen1, textvariable=password, show="**")
    password_entry.pack()
    Label(screen1, text="", font=("Times New Roman",)).pack()
    Button(screen1, text="Register", width=10, height=1, command=register_user).pack()
    Button(screen1, text="Back", width=10, height=1, command=main_screen).pack()
    Label(screen1, text="Partial Face Detection", bg="steelblue", fg="black", width=250, height=5,font=("Arial Bold", 16)).pack(side=BOTTOM)


def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_entry1.delete(0, END)
    password_entry1.delete(0, END)

    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_successfully()
        else:
            Please_enter_correct_password()
    else:
        username_not_found()


def login():
    global screen2
    screen2 = Toplevel(screen)
    screen2.title("Login")
    screen2.geometry("1350x728")
    Label(screen2, text="Please Enter Details To Login",fg="black",bg="steelblue", width=400, height=5, font=("Arial Bold",16)).pack()
    global username_verify
    global password_verify
    username_verify = StringVar()
    password_verify = StringVar()
    global username_entry1
    global password_entry1
    Label(screen2, text="",bg="steelblue",fg="black", width=5, height=100, font=("Arial Bold",16)).pack(side=RIGHT)
    Label(screen2, text="",bg="steelblue",fg="black", width=5, height=100, font=("Arial Bold",16)).pack(side=LEFT)
    Label(screen2, text="Username", font=("Times New Roman",)).pack()
    username_entry1 = Entry(screen2, textvariable=username_verify)
    username_entry1.pack()
    Label(screen2, text="").pack()
    Label(screen2, text="Password *", font=("Times New Roman",)).pack()
    password_entry1 = Entry(screen2, textvariable=password_verify, show="**")
    password_entry1.pack()
    Button(screen2, text="Forget password",bg="white", fg="red", width=12, height=1, command=Forget_Password).pack()
    Button(screen2, text="Login", bg="white", fg="red",font=("Times New Roman",), width=6, height=1, command=login_verify).pack()
    Label(screen2, text="Partial Face Detection",bg="steelblue",fg="black", width=250, height=5, font=("Arial Bold",16)).pack(side=BOTTOM)


def user_recovery_verify():
    username2 = username_verify.get()
    username_entry2.delete(0, END)

    list_of_files = os.listdir()
    if username2 in list_of_files:
        file2 = open(username_info, "r")
        verify = file2.read().splitlines()
        Password_recovery()
    else:
        username_not_found()


def Forget_Password():
    global screen9
    screen9 = Toplevel(screen)
    screen9.title("Password Recovery")
    screen9.geometry("300x250")
    global username_entry2
    global username2
    username2 = StringVar()
    global username2_verify
    username2_verify = StringVar()
    Label(screen9, text="Please Enter Username").pack()
    username_entry2 = Entry(screen9, textvariable=username2_verify)
    username_entry2.pack()
    Button(screen9, text="Submit", fg="black", bg="grey", width=10, height=2,command=user_recovery_verify).pack()


def new_password_save():
    global password2
    global password3
    password2 = StringVar()
    password3 = StringVar()
    password1_info = password2.get()
    password2_info = password3.get()
    global password_entry2
    global password_entry3
    password2 = password_verify.get()
    password_entry2.delete(0, END)

    list_of_files = os.listdir()
    if password2 in list_of_files:
        file = open(username_info, "r")
        verify = file.read().splitlines()
        if password1_info == password2_info:
            file = open(username_info, "w")
            file.write(password_info + "\n")
        else:
            New_password_not_same()
    else:
        Username_is_not_available()

    username_entry.delete(0, END)
    password_entry.delete(0, END)


def Password_recovery():
    global screen10

    password2 = StringVar()
    screen10 = Toplevel(screen)
    screen10.title("Password Recovery")
    screen10.geometry("300x250")
    Label(screen10, text="Old password").pack()
    password_entry2 = Entry(screen10, textvariable=password2,show="**")
    password_entry2.pack()
    Label(screen10, text="New password").pack()
    password_entry3 = Entry(screen10, textvariable=password3,show="**")
    password_entry3.pack()
    password_entry3.pack()
    Label(screen10, text="Re-write New password").pack()
    password_entry3 = Entry(screen10)
    password_entry3.pack()
    Button(screen10, text="Confirm", bg="black", fg="white", width=9, height=1,command=new_password_save).pack()


def New_password_not_same():
    global screen12
    screen12 = Toplevel(screen)
    screen12 = Tk()
    screen12.title("My App")
    screen.geometry("300x200")
    Label(screen12, text="Please enter correct password", bg="indigo", fg="white", width=8, height=1).pack()
    Button(screen12, text="ok", bg="black", fg="white", width=4, height=1, command=Password_recovery).pack()


def please_enter_correct_password():
    global screen11
    screen11 = Toplevel(screen)
    screen11 = Tk()
    screen11.title("My App")
    screen11.geometry("300x200")
    Label(screen11, text="Please enter correct password", bg="indigo", fg="white", width=8, height=1).pack()
    Button(screen11, text="ok", bg="black", fg="white", width=4, height=1, command=Password_recovery).pack()


main_screen()
