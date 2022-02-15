import pymongo
import datetime
import pprint
#database configurations
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
mongoDb = mongoClient["queue_analytics"]

#collections in the mongodb
mongoCashierCollection = mongoDb["cashierdata"]
mongoQueueCollection = mongoDb["queue_data"]
mongoVideoCollection = mongoDb["videodata"]


# if mongoQueueCollection.find({'Person ID': 5}).count() > 0 :
#     print("person has already entered")


def insertCashierData(cashierNo, currentTime, cashierAvailability):
    print("ok")
    
    # # getting the last entered time for the cashier number   
    # last_entered_time = mongoCashierCollection.find_one({'Cashier No': cashierNo}, sort=[('_id', pymongo.DESCENDING)])
    # last_entered_time = last_entered_time["Time"]
    # c_time = currentTime
    # difference_time = (c_time - last_entered_time).total_seconds()
    # if difference_time > 10:
    #     #cashier data dictionary
    #     cashier_data = {"Cashier No": cashierNo, "Time" : currentTime, "Cashier is Available": cashierAvailability}
    #     #insert cashier data
    #     output = mongoCashierCollection.insert_one(cashier_data)
    #     #after insert
    #     print("Cashier Data Inserted Successfully")
    #     print("Insert ID")
    #     print(output.inserted_id)
    # else:
    #     print("do nothing")

def insertQueueLength(timeNow, peopleInsideQueue, videoID):
    print("ok")

    ## Waiting Time From QueueData
    # agg_result= mongoQueueCollection.aggregate(
    # [
    #     {
    #         "$match": {"hasLeft" : "0" , "videoID" : videoID}
    #     },
    #     {
    #     "$group" : 
    #         {"_id" : "$videoID", 
    #         "Wait" : {"$sum" : "$Wait"}
    #         }}
    # ])
    # print("waiting time total")

    # for i in agg_result:
    #     print(i)
    #     waitTime = i["Wait"]
    # print(waitTime)
    # averageWaitingTime = waitTime / peopleInsideQueue

    # #Enter Analytics For Video Data 

    # last_entered_record = mongoVideoCollection.find_one({'videoID': videoID}, sort=[('_id', pymongo.DESCENDING)])
    # if last_entered_record != None:


    #     last_entered_time = last_entered_record["Time"]
    #     last_entered_second = last_entered_record["timeInSecond"]

        
    #     cTime = timeNow
    #     differenceTime = (cTime - last_entered_time).total_seconds()
    
    #     if differenceTime > 10:
    #         last_entered_second = int(last_entered_second)
    #         last_entered_second +=1


    #         queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "videoID": videoID, "timeInSecond": str(last_entered_second), "averageWaitingTime":averageWaitingTime}

    #         #insert queue length data
    #         output = mongoVideoCollection.insert_one(queue_length_data)
    #     else: 
    #         print("ok")
    # else:

    #     print("its empty")
    #     print("insert queue length")
    #     queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "videoID": videoID, "timeInSecond": "0", "averageWaitingTime" : averageWaitingTime}

    #     #insert queue length data
    #     output = mongoVideoCollection.insert_one(queue_length_data)
        
    #     #after insert
    #     print("Queue data inserted successfully")


# def insertQueueLength(timeNow, peopleInsideQueue, videoID):
#     print("ok")

#     # getting the last entered time 
#     # change last entered time
#     last_entered_record = mongoVideoCollection.find_one({'videoID': videoID}, sort=[('_id', pymongo.DESCENDING)])
#     if last_entered_record != None:


#         last_entered_time = last_entered_record["Time"]
#         last_entered_second = last_entered_record["timeInSecond"]

        
#         cTime = timeNow
#         differenceTime = (cTime - last_entered_time).total_seconds()
    
#         if differenceTime > 10:
#             last_entered_second = int(last_entered_second)
#             last_entered_second +=1


#             queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "videoID": videoID, "timeInSecond": str(last_entered_second)}

#             #insert queue length data
#             output = mongoVideoCollection.insert_one(queue_length_data)
#         else: 
#             print("ok")
#     else:

#         print("its empty")
#         print("insert queue length")
#         queue_length_data = {"Time": timeNow, "qlength" : peopleInsideQueue, "videoID": videoID, "timeInSecond": "0"}

#         #insert queue length data
#         output = mongoVideoCollection.insert_one(queue_length_data)
        
#         #after insert
#         print("Queue data inserted successfully")
        
 



def insertQueueData(personID, inTime):

    # x = mongoQueueCollection.find_one({'PersonID': personID})
    

    # if x != None:
    #     print("person has already entered--------------------------------------")
    #     last_updated_time = mongoQueueCollection.find_one({'PersonID':personID})
    #     mongoID = last_updated_time["_id"]
    #     last_updated_time = last_updated_time["InTime"]
    #     currentTime = inTime
    #     difference = (currentTime - last_updated_time).total_seconds()

    #     myquery = {"PersonID": personID}
    #     newvalues = {"$set":{"Wait":difference}}

    #     mongoQueueCollection.update_one(myquery,newvalues)

    # else:
    #     #cashier data dictionary
    #     customer_data = {"PersonID": personID, "InTime" : inTime, "Wait":0}
    #     #insert cashier data
    #     output = mongoQueueCollection.insert_one(customer_data)


    # # after insert
    print("Customer Data Inserted Successfully")
    print("Insert ID")

    

    

