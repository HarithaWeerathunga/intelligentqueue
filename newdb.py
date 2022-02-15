import pymongo
import datetime
import pprint
#database configurations
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
mongoDb = mongoClient["analytics"]

#collections in the mongodb
mongoCashierCollection = mongoDb["cashierdata"]
# mongoQueueCollection = mongoDb["queue_data"]
# mongoVideoCollection = mongoDb["video_data"]

# Testin purpose
mongoQueueCollection = mongoDb["camera1jun6_q1"]
mongoVideoCollection = mongoDb["camera1jun6_v1"]



def insertTrackingData(personID, inTime, videoID):

    x = mongoQueueCollection.find_one({'PersonID': personID})
    

    if x != None:
        print("person has already entered--------------------------------------")
        last_updated_time = mongoQueueCollection.find_one({'PersonID':personID, 'videoID': videoID})
        print("last updated time")
        print(x)
        print(last_updated_time)
        last_updated_time = last_updated_time["InTime"]
        currentTime = inTime
        difference = (currentTime - last_updated_time).total_seconds()

        myquery = {"PersonID": personID}
        newvalues = {"$set":{"Wait":difference}}

        mongoQueueCollection.update_one(myquery,newvalues)

    else:
        #customer data dictionary
        customer_data = {"PersonID": personID, "InTime" : inTime, "Wait":0, "videoID": videoID, "hasLeft": "0", "vid": str(videoID)}
        #insert customer data
        output = mongoQueueCollection.insert_one(customer_data)


    # # # # after insert
    print("Customer Data Inserted Successfully")
    print("Insert ID")

def updateExitStatus(trackid, videoID):
    myquery = {"videoID" : videoID, "PersonID" : trackid}
    newvalues = {"$set": {"hasLeft" : "1"}}
    mongoQueueCollection.update_one(myquery,newvalues)
    print("updated exit status")

def updateEntranceStatus(trackid, videoID):
    myquery = {"videoID" : videoID, "PersonID" : trackid}
    newvalues = {"$set": {"hasLeft" : "0"}}
    mongoQueueCollection.update_one(myquery,newvalues)
    print("updated exit status")


def insertQueueAnalytics(timeNow, peopleInsideQueue,cashierAvailability,videoID):

    if cashierAvailability > 0:
        cashierAvailability = "Present"
    else:
        cashierAvailability = "Absent"

    x = mongoQueueCollection.find_one({'videoID': videoID})

    if x != None:
        # Waiting time from tracking data
        agg_result= mongoQueueCollection.aggregate(
        [
            {
                "$match": {"hasLeft" : "0" , "videoID" : videoID}
            },
            {
            "$group" : 
                {"_id" : "$videoID", 
                "Wait" : {"$sum" : "$Wait"}
                }}
        ])
        print("waiting time total")

        for i in agg_result:
            print(i)
            waitTime = i["Wait"]
    else :
        print("ok")
        waitTime = 0

    if peopleInsideQueue == 0:
        waitTime = 0
        averageWaitingTime = 0
    else:
        averageWaitingTime = waitTime / peopleInsideQueue

    averageWaitingTime = round(averageWaitingTime,2)
    #Enter Analytics For Video Data 

    last_entered_record = mongoVideoCollection.find_one({'videoID': videoID}, sort=[('_id', pymongo.DESCENDING)])
    if last_entered_record != None:


        last_entered_time = last_entered_record["Time"]
        last_entered_second = last_entered_record["timeInSecond"]

        
        cTime = timeNow
        differenceTime = (cTime - last_entered_time).total_seconds()
        ##1.3 best right now
        ##camera 8 > 1.50
        # camera 1 > 1.220
        # camera 4 > 1.20
        print("DIFFERENCE")
        print(differenceTime)
        if differenceTime > 1.220 :
            last_entered_second = int(last_entered_second)
            last_entered_second +=1


            queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "cashierAvailability" : cashierAvailability,"videoID": videoID, "timeInSecond": str(last_entered_second), "averageWaitingTime":averageWaitingTime, "vid":str(videoID)}

            #insert queue length data
            output = mongoVideoCollection.insert_one(queue_length_data)
        else: 
            print("ok")
    else:

        print("its empty")
        print("insert queue length")
        queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "cashierAvailability" : cashierAvailability,"videoID": videoID, "timeInSecond": "0", "averageWaitingTime" : averageWaitingTime, "vid": str(videoID)}

        #insert queue length data
        output = mongoVideoCollection.insert_one(queue_length_data)
        
        #after insert
        print("Queue analytics inserted successfully")

