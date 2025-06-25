#通信部分代码
import serial,sys

class Connector:
    def __init__(self,bandrate=115200):
        print(serial.VERSION)
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', bandrate, timeout=5)
            self.ser.bytesize = 8
            print(self.ser)
        except:
            print("通信出错")
            self.ser=None
            return None




    def Port_receive(self):#读取数据
        return self.ser.readline()

    def Port_send(self,data):
        return self.ser.write(data)

if __name__=='__main__':
    c=Connector()
    c.Port_send("debug")