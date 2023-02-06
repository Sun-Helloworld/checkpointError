import os
import time
print(os.getpid())
if os.path.exists("/root/pytorch/gpu-t/myd"):
    os.rmdir("myd")
    os.mkdir("myd1")
    while True:
        time.sleep(1)
        if os.path.exists("myd2"):
            os.mkdir("myd3")
            print(2)
            break



    
