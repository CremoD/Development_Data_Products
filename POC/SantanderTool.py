import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tkinter as tk
from tkinter import ttk
import SharingAnalysis as sa
import SharingPrediction as sp


def change_label(selected):
  result_label.configure(text="Past data analysis: " + selected)
	#figure = sa.grid_plots(selected)
	#chart_type = FigureCanvasTkAgg(figure, frame_image)
	#chart_type.get_tk_widget().pack()
  if selected != 'All':
    sa.grid_plots(selected)
  else:
    sa.grid_plots('total')
  im = tk.PhotoImage(file="plots/analysis.png")
  image_label.configure(image=im)
  image_label.image=im


def change_label2(selected, forecast_df):
  result_label.configure(text="Future shares prediction: " + selected) 

  sp.prediction_plots(forecast_df, selected)
  im = tk.PhotoImage(file="plots/prediction.png")
  image_label.configure(image=im)
  image_label.image=im




# scrollable class
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas,width=800)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")



df = pd.read_csv('weather_bike_usage.csv')
stations = df['StartStation Name'].unique()
stations = list(stations)
stations2 = stations.copy()
stations.insert(0,"All")


# Application
root = tk.Tk()
root.geometry('1200x800')
root.title("Santander cycles management")

logo = tk.PhotoImage(file="icons/santander.png")

# icon and explanation
explanation = """Manage previous bike usage data 
and predict future sharings based
on day and weather information.
       4 Jan 2012 - 3 Jan 2015"""
w = tk.Label(root, 
          justify=tk.LEFT,
          compound = tk.LEFT,
          padx = 30, 
          text=explanation, 
          image=logo,
          font=('arial',20,'bold')).pack(side="top")


# first functionality, visualize previous data plots
frame1 = tk.Frame(root)
heading1 = tk.Label(frame1, text="Analyse previous data for",pady=20, font=('arial',15,'bold'))
heading1.pack(side=tk.LEFT)
combo1 = ttk.Combobox(frame1, values=stations)
combo1.current(0)
combo1.pack(side=tk.LEFT)
button1 = tk.Button(frame1, text="Analyse", padx = 30, command = lambda: change_label(combo1.get()))
button1.pack(side = tk.RIGHT)
frame1.pack()



# second functionality, predict
frame2 = tk.Frame(root)
heading2 = tk.Label(frame2, text="Predict future shares for ",pady=20, font=('arial',15,'bold'))
heading2.pack(side = tk.LEFT)
combo2 = ttk.Combobox(frame2, values=stations2)
forecast_df = sp.forecasts_creation(stations2)
combo2.pack(side = tk.LEFT)
combo2.current(0)
button2 = tk.Button(frame2, text="Predict", padx = 30, command = lambda: change_label2(combo2.get(), forecast_df))
button2.pack(side = tk.RIGHT)
frame2.pack()


# Heading for results
result_label=tk.Label(root, font=('arial',20,'bold'), pady = 20)
result_label.pack()

# resulted image
frame_image = tk.Frame(root,width=800)




canvas = tk.Canvas(frame_image, width = 1100, height = 450)
scrollbar = ttk.Scrollbar(frame_image, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas, width = 1000, height = 500)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


image_label = tk.Label(scrollable_frame)
#frame_image.pack()
image_label.pack()

frame_image.pack()
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")





root.mainloop()