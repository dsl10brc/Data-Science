import logging
import urllib2
logging.basicConfig(filename='example.log',level=logging.DEBUG,format='%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s')
def main():
    try:
        urls = "http://www.simplyhired.com/a/job-detaw/jobkey-67b4efe169eee7b2cf2ed47d49b1845070ea37/rid-racliggzfyjgqwfzrlvnqyjtcserhrri/cjp-3/pub_id-1002"
        site = urllib2.urlopen(urls).read()
        mathfail = 1/0
    except Exception, e:
        logging.exception(str(e))

main()

