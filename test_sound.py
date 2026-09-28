from pygame import mixer
import time

mixer.init()
s = mixer.Sound("alarm.wav")
s.set_volume(1.0)
s.play()
time.sleep(3)
print("done")
