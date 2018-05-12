import tkinter as tk
import cv2
from skimage.measure import compare_ssim
import imutils
import datetime
from PIL import Image, ImageTk
import asyncio

class EVA(tk.Frame):
    def __init__(self, master=None, loop=None):
        super().__init__(master)
        self.loop = loop
        self.pack()

        # Program settings
        self.debug = False
        self.sensitivity = 0.94
        self.save = False

        # webcam settings
        self.width, self.heigth = 800, 600

        # Capture webcam, change 0 to any device input
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heigth)

        # Live video stream frame
        # Attach to parent root
        self.vid_frame = tk.Label(master)
        self.vid_frame.pack()

        # Debug output
        self.score = None
        self.score_history = []


        # Debug frame
        self.debug_frame = tk.Frame(self.master, width=60, height=80)
        self.debug_frame.pack(fill="both", expand=True)

        # Buttons
        self.btn_frame = tk.Frame(self.master, width=60, height=80)
        self.btn_frame.pack(fill="both", expand=True)

        # Consistent GUI size
        self.debug_frame.grid_propagate(False)
        self.btn_frame.grid_propagate(True)

        # Implement stretchability
        self.debug_frame.grid_rowconfigure(0, weight=1)
        self.debug_frame.grid_columnconfigure(0, weight=1)
        self.btn_frame.grid_rowconfigure(0, weight=1)
        self.btn_frame.grid_columnconfigure(0, weight=1)


    def create_widgets(self):
        self.winfo_toplevel().title("EVA - Webcam security")
        # Quit button
        self.quit = tk.Button(master=self.master, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="right")

        # Text part
        self.saved_output = tk.Text(self.master, borderwidth=3, relief="sunken", width=25, height=1)
        self.saved_output.config(font=("consolas", 12), undo=True, wrap="word")
        self.saved_output.pack(side="right")

        # Save movement button
        self.save_btn = tk.Button(
            master=self.btn_frame,
            text="Save to disk",
            command=self.toggle_save_img_to_disk)
        self.save_btn.pack(side="right")

        # Dev button
        self.dev_btn = tk.Button(
            master=self.btn_frame,
            text="Debug",
            command=self.toggle_debug)
        self.dev_btn.pack(side="right")

        # Sensitivity slider
        self.slider = tk.Scale(self.btn_frame,
                               command=self.toggle_sensitivity,
                               variable = self.sensitivity,
                               from_ = 1.,
                               to = 0.90,
                               length = 150,
                               digits = 3,
                               resolution = 0.01,
                               orient="horizontal",
                               label="Sensitivity motion detector")
        self.slider.pack(side="right")

    def toggle_save_img_to_disk(self):
        # Check bool for saving to disk
        if self.save == False:
            self.save = True
            self.save_btn.config(text='On')
        else:
            self.save = False
            self.save_btn.config(text='Save to disk')
        print(f"Save to disk is :{self.save}")
        return

    def toggle_debug(self):
        # Check bool for debug to disk
        if self.debug == False:
            self.debug = True
            self.dev_btn.config(text='On')
        else:
            self.debug = False
            self.dev_btn.config(text='Debug')
            self.debug_output.delete("1.0",tk.END)
            cv2.destroyWindow("Frame: B")
            cv2.destroyWindow("Diff")
            cv2.destroyWindow("Thresh")
        print(f"Debug is toggled :{self.debug}")
        return

    def toggle_sensitivity(self, slider_value):
        if slider_value:
            self.sensitivity_current = float(slider_value)
            if self.sensitivity_current != self.sensitivity:
                # prevent the user to go lower than 0.94. Beyond this number no motion will be detected
                if self.sensitivity_current >= 0.90 and self.sensitivity_current <= 0.94:
                    self.sensitivity = 0.94
                else:
                    self.sensitivity = self.sensitivity_current
                print(f'Current motion sensitivity: {self.sensitivity_current}\nPrevious motion sensitivity: {self.sensitivity}')

    """
    Example on how to show webcam in tkinter:
    ------------------------------------------
    def show_frame(self):
        _, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.vid_frame.imgtk = imgtk
        self.vid_frame.configure(image=imgtk)
        self.vid_frame.after(10, self.show_frame)
    """
    def show_frame(self):
        # Capture frame-by-frame
        frame_a = self.capture.read()[1]
        frame_a = cv2.flip(frame_a, 1)
        # Gray out our first frame
        grayA = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        # Capture second frame
        frame_b = self.capture.read()[1]
        frame_b = cv2.flip(frame_b, 1)
        # Gray out second frame
        grayB = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # compare Structural Similarity Index (SSIM) between the two frames
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        self.score = score
        # debug print
        if self.debug:
            # Debug output
            # Text part
            self.debug_output = tk.Text(self.debug_frame, borderwidth=3, relief="sunken")
            self.debug_output.config(font=("consolas", 12), undo=True, wrap="word")
            self.debug_output.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
            self.score_history.append(score)
            # Calculate average
            total_entity = len(self.score_history)
            total_score = 0

            for i in self.score_history:
                total_score += i
            average = total_score / total_entity
            self.debug_output.insert(tk.INSERT, f'Current SSIM: {score}{" "*int(20-len(str(score)))}'
                                                f'Average SSIM: {average}\n'
                                                f'Minimal SSIM: {min(self.score_history)}{" "*int(20-len(str(score)))}'
                                                f'Max     SSIM: {max(self.score_history)}')

            #create scrollbar for debug output
            self.debug_scrollbar = tk.Scrollbar(self.debug_frame, command=self.debug_output.yview)
            self.debug_scrollbar.grid(row=0, column=1, sticky="nsew")
            self.debug_output["yscrollcommand"] = self.debug_scrollbar.set
            self.debug_scrollbar.config(command=self.debug_output.yview)

            print("SSIM: {}".format(self.score))

        # If image score is lower or equal to sensitivity to take action.
        # This means the difference between the two frames are major enough
        # To call it a detection of motion.
        if score <= self.sensitivity:
            # threshold the difference in the frames, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            # compute the bounding box of the contour and then draw the
            # bounding box on both input frames to represent where the two
            # frames differ
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame_a, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame_b, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # # Display the frame A
        # cv2.imshow("Frame: A", frame_a)

        cv2image = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.vid_frame.imgtk = imgtk
        self.vid_frame.configure(image=imgtk)
        self.vid_frame.after(10, self.show_frame)

        if score <= self.sensitivity:
            # Create a picture and store it in the img folder.
            # The file is named the current date and time.
            if self.save:
                cv2.imwrite('img/{}.png'.format(datetime.datetime.now().strftime("%d-%B-%Y-%I%M%S%p")), frame_a)

            # If debug modus
            if self.debug:
                # Display debug frames
                cv2.imshow("Frame: B", frame_b)
                cv2.imshow("Diff", diff)
                cv2.imshow("Thresh", thresh)

            self.saved_output.delete("1.0",tk.END)
            self.saved_output.insert(tk.INSERT, "Motion detected")
        else:
            # Text part
            self.saved_output.delete("1.0", tk.END)
            self.saved_output.insert(tk.INSERT, "No Motion")

if __name__ == '__main__':
    # Define loop
    loop = asyncio.get_event_loop()
    # Define parent root
    root = tk.Tk()
    root.resizable(width=False, height=False)
    # Define EVAs buttons
    EVA = EVA(master=root, loop=loop)
    loop.create_task(EVA.create_widgets())
    loop.create_task(EVA.show_frame())
    # Start root loop
    loop.create_task(root.mainloop())