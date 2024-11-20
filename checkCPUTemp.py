import os
import time

def get_cpu_temp_linux():

    while(True):
        try:
            outstr = ""
            cnt = 0
            for i in range(4):
                temp_file = f"/sys/class/thermal/thermal_zone{cnt}/temp"
                if os.path.isfile(temp_file):
                    with open(temp_file, "r") as f:
                        temp = int(f.read()) / 1000.0  # convert from millidegree
                        outstr+=f"[TZ-{cnt}={temp} degC] "
                cnt+=1

            print(f"{outstr}")
            outstr = None
            cnt = None
            time.sleep(1)

        except KeyboardInterrupt:
            try:
                input('pausing monitor, <cr> to continue, ^C to end..')
            except KeyboardInterrupt:
                print('\n')
                return


if __name__ == '__main__':
    get_cpu_temp_linux()
