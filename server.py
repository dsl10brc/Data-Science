import zerorpc, sys
from threading import Thread
from Simplyhired import trg
from Indeed import trg1

class HelloRPC(object):


	def hello(self, message):
		print message
		if not message:
			print wait
		else:
			print "querystring is  "
			queryString= message.split(',')
			print queryString
			print "Requirements of Crawler" ,message
			print queryString
			#queryString[3],
			#trg(queryString[3],queryString[0],queryString[1],queryString[2]) 
			thread1 = Thread(target=trg,args=(queryString[3],queryString[4],queryString[5],queryString[0],queryString[1],queryString[2]))
			thread2 = Thread(target=trg1,args=(queryString[3],queryString[4],queryString[5],queryString[0],queryString[1],queryString[2]))
			thread1.start()
			thread2.start()

def main():
	while not exitapp:	
		s = zerorpc.Server(HelloRPC())
		s.bind("tcp://0.0.0.0:4242")
		s.run()
		#sys.stdin.readlines()

exitapp = False

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exitapp = True
        raise	
		

