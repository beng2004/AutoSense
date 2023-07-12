
from pytube import YouTube 
  
# where to save 
SAVE_PATH = "C:/Users/benjamin.guerrieri/Documents/AutoSenseBackup" #to_do 
  
# link of the video to be downloaded 
link='https://www.youtube.com/watch?v=e_WBuBqS9h8'
  
yt = YouTube(link)  

# try:
yt.streams.filter(progressive = True, 
file_extension = "mp4").first().download(output_path = SAVE_PATH, 
filename = "TrafficVideo3.MP4")
# except:
#     print("Some Error!")
# print('Task Completed!')
