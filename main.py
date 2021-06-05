import cv2
import numpy as np
import pdb
import math
import time
from copy import copy, deepcopy

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    
    def __repr__(self):
        return f"[x : {self.x}, y : {self.y}]"

    def __str__(self):
        return f"x : {self.x}, y : {self.y}"

class Circle:
    def __init__(self, point, radius, name):
        self.name = str(name)
        self.point = point
        self.radius = int(radius)

    def __repr__(self):
        return f"nameCircle : {self.name}, circleCenter : {self.point}, radius : {self.radius}"
    
    def __str__(self):
        return f"nameCircle : {self.name}, circleCenter : {self.point}, radius : {self.radius}"

class Event:
    def __init__(self, circle, pointEvent, typeEvent):
        self.name = circle.name + typeEvent
        self.circle = circle
        self.pointEvent = pointEvent
        self.typeEvent = typeEvent

    def __repr__(self):
        return f"nameEvent : {self.name}, circle : {self.circle}, pointEvent : {self.pointEvent}, typeEvent : {self.typeEvent}"
    
    def __str__(self):
        return f"nameEvent : {self.name}, circle : {self.circle}, pointEvent : {self.pointEvent}, typeEvent : {self.typeEvent}"

def parseInput(file1):
    Lines = file1.readlines()

    data = Lines[0].split()
    x = Point(data[0], data[1])
    data = Lines[1].split()
    y = Point(data[0], data[1])

    count = 0
    circles = []
    for line in Lines[2:]:
        data = line.split()
        circles.append(Circle(Point(data[0], data[1]), data[2], count))
        print("Point {}: {}".format(count, line.strip()))
        count += 1
    
    return x, y, circles   

def draw(x,y,circles):
    scaleFactor = 50

    background = np.zeros((x.y * scaleFactor, y.y * scaleFactor, 3), dtype=np.uint8)

    # Line thickness of 2 px
    thickness = 2
    
    # fontScale
    fontScale = 1

    # Draw a circle with blue line borders of thickness of 2 px
    for circle in circles:
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        position = (circle.point.x * scaleFactor, circle.point.y * scaleFactor)
        cv2.circle(background, position, circle.radius * scaleFactor, color, thickness)
        cv2.circle(background, position, 1, color, 5)
        cv2.putText(background, circle.name, (position[0]+10,position[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 4, cv2.LINE_AA)

    cv2.imshow("background", background)
    cv2.waitKey(0)

    for i in range(len(circles)):
        for j in range(len(circles)):
            intersectionPoints = computeIntersection(circles[i], circles[j], scaleFactor)

            if intersectionPoints is not None:
                for p in intersectionPoints:
                    background = cv2.circle(background, (p.x, p.y), 1, (0,0,255), 10)

    cv2.imshow("background", background)
    cv2.waitKey(0)

def screenShotPlaneSweep(x, y, circles, event, intersection = None, animation = True):
    
    if animation:            
        scaleFactor = 50
        background = np.zeros((x.y * scaleFactor, y.y * scaleFactor, 3), dtype=np.uint8)

        # Line thickness of 2 px
        thickness = 2
        
        # fontScale
        fontScale = 1

        # Draw a circle with blue line borders of thickness of 2 px
        for circle in circles:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            position = (circle.point.x * scaleFactor, circle.point.y * scaleFactor)
            cv2.circle(background, position, circle.radius * scaleFactor, color, thickness)
            cv2.circle(background, position, 1, color, 5)
            cv2.putText(background, circle.name, (position[0]+10,position[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 4, cv2.LINE_AA)

        cv2.line(background, (event.pointEvent * scaleFactor, 0), (event.pointEvent * scaleFactor, x.y * scaleFactor), (0, 255, 0), thickness=2)
        if event.pointEvent * scaleFactor <= int(x.y * scaleFactor / 2):
            pos = (event.pointEvent * scaleFactor + 20, int((x.y * scaleFactor) / 2))
        else:
            pos = (event.pointEvent * scaleFactor - 130, int((x.y * scaleFactor) / 2))
        cv2.putText(background, event.typeEvent, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), 4, cv2.LINE_AA)
        cv2.imshow("background", background)
        cv2.waitKey(0)

        if intersection is not None:
            for spa in intersection:
                for p in intersection[spa]["points"]:
                    background = cv2.circle(background, (p.x, p.y), 1, (0,0,255), 10)
            
            cv2.imshow("background", background)
            cv2.waitKey(0)


def computeIntersection(circle_a, circle_b, scaleFactor):
    x0, y0, r0 = circle_a.point.x * scaleFactor, circle_a.point.y * scaleFactor,  circle_a.radius * scaleFactor
    x1, y1, r1 = circle_b.point.x * scaleFactor, circle_b.point.y * scaleFactor, circle_b.radius * scaleFactor

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
   
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return [Point(x3, y3), Point(x4, y4)]

def pretty(d, indent=0):       
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def findIntersection(sweepline, index, circles, scaleFactor = 50):
    intersectionPointsUp = []
    intersectionPointsBelow = []

    if len(sweepline) > 1:
        up = sweepline[index]["up"]
        below = sweepline[index]["below"]

        # if index == 2:
        #     pdb.set_trace()

        if up is not None:
            intersectionPointsUp = computeIntersection(circles[index], circles[up], scaleFactor)
            if intersectionPointsUp is None:
                intersectionPointsUp = []

        if below is not None:
            intersectionPointsBelow = computeIntersection(circles[index], circles[below], scaleFactor)
            if intersectionPointsBelow is None:
                intersectionPointsBelow = []

    return [intersectionPointsUp, intersectionPointsBelow]


def deleteElementFromSweepLine(sweepline, index):
    up = sweepline[index]["up"]
    below = sweepline[index]["below"]

    if up is not None:
        if below is not None:
            sweepline[up]["below"] = below
        else:
            sweepline[up]["below"] = None

    if below is not None:
        if up is not None:
            sweepline[below]["up"] = up
        else:
            sweepline[below]["up"] = None

    del sweepline[index]
    
rootIndex = 0
def updateSweepLine(sweepline, currentEvent, circles):
    if len(sweepline) == 0:
        global rootIndex
        rootIndex = int(currentEvent["obj"].circle.name)
        sweepline[int(currentEvent["obj"].circle.name)] = {
            "up" : None,
            "me" : rootIndex,
            "below" : None,
        }
    else: 
        currentCircle = sweepline[rootIndex]
        eventCircleY = currentEvent["obj"].circle.point.y

        while True:

            # If the sweep line has only 1 element
            if currentCircle["up"] is None and currentCircle["below"] is None:
                swapCircleMe = circles[currentCircle["me"]].point.y

                if eventCircleY > swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : None,
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                else:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["me"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : None,
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                
                break # we exit immediately

            # If the currentCircle of the sweep line has a None below circle
            elif currentCircle["up"] is None:
                swapCircleMe = circles[currentCircle["me"]].point.y
                swapCircleYBelow = circles[currentCircle["below"]].point.y

                if swapCircleYBelow <= eventCircleY <= swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["me"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["below"],
                    }

                    tmp = sweepline[currentCircle["me"]]["below"]
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["up"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY > swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : None,
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["me"],
                    }

                    # update the current sweep circle that is below the new circle
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                    break

                else:
                    # we look at below
                    currentCircle = sweepline[currentCircle["below"]]

            # If the currentCircle of the sweep line has a None up circle
            elif currentCircle["below"] is None:
                swapCircleYUp = circles[currentCircle["up"]].point.y
                swapCircleMe = circles[currentCircle["me"]].point.y

                if swapCircleMe <= eventCircleY <= swapCircleYUp:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["up"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)

                    sweepline[currentCircle["up"]]["below"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY < swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["me"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : None,
                    }

                    # update the current sweep circle that is up
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    break

                else:
                    # we look at up
                    currentCircle = sweepline[currentCircle["up"]]
                
            else:
                swapCircleYUp = circles[currentCircle["up"]].point.y
                swapCircleMe = circles[currentCircle["me"]].point.y
                swapCircleYBelow = circles[currentCircle["below"]].point.y

                if swapCircleMe <= eventCircleY <= swapCircleYUp:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["up"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                    sweepline[currentCircle["up"]]["below"] = int(currentEvent["obj"].circle.name)
                    break

                elif swapCircleYBelow <= eventCircleY <= swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = { 
                        "up" : currentCircle["me"],
                        "me" : int(currentEvent["obj"].circle.name),
                        "below" : currentCircle["below"],
                    }

                    tmp = sweepline[currentCircle["me"]]["below"]
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["up"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY > swapCircleYUp:
                    currentCircle = sweepline[currentCircle["up"]]
                    
                elif eventCircleY < swapCircleYBelow:
                    currentCircle = sweepline[currentCircle["below"]]

def planeSweepPacking(x, y, circles, animation):
    events = []
    for circle in circles:
        events.append(Event(circle, circle.point.x - circle.radius, "LEFT"))
        events.append(Event(circle, circle.point.x + circle.radius, "RIGHT"))

    print("\nNot ordered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")

    # A way that can be fastest, especially if your list has a lot 
    # of records, is to use operator.attrgetter("count")
    try: import operator
    except ImportError: keyfun= lambda x: x.count # use a lambda if no operator module
    else: keyfun= operator.attrgetter("pointEvent") # use operator since it's faster than lambda

    events.sort(key=keyfun, reverse=False) # sort in-place

    print("\nOrdered events:")  
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")
    print("\n") 

    '''
    # example of eventsQueue
    eventsQueue = {
        "event1" : {
            "obj" : event1,
            "before" : "event0",
            "after" : "event2",
        },

        "event2" :{
            "obj" : event2,
            "before" : "event1",
            "after" : "event3",
        }
    }

    # example of sweepline
    sweepline = {
        0 : {               # name of this circle
            "up" : None,    # name of upper circle
            "me" : 0,       # name of this circle
            "below" : 2,    # name of lower circle
        },

        2 :{
            "up" : 0,
            "me" : 2,
            "below" : 1,
        }

        1 :{
            "up" : 2,
            "me" : 1,
            "below" : None,
        }
    }
    '''

    sweepline = {}

    # We set the priority through a dictionary of events
    eventsQueue = {}
    eventsQueue[events[0].name] = {"obj": events[0], "before" : None, "after" : events[1].name}

    for i in range(1, len(events[1::])):
        eventsQueue[events[i].name] = {"obj": events[i], "before" : events[i-1].name, "after" : events[i+1].name}

    eventsQueue[events[-1].name] = {"obj": events[-1], "before" : events[-2].name, "after" : None}

    # Swap line among the events
    count = 0
    currentEvent = eventsQueue[events[0].name]
    while True:
        print(f"Iteration: {count}")
        print(f'Name current obj: {currentEvent["obj"].name}')
        print(eventsQueue)
        # pretty(eventsQueue) # pretty view of data
        print(f"Length eventsQueue: {len(eventsQueue)} \n")

        # inside the rectangle R
        if x.x <= currentEvent["obj"].pointEvent <= x.y: 

            # LEFT EVENT
            if currentEvent["obj"].typeEvent == "LEFT":
                
                if int(currentEvent["obj"].circle.name) not in sweepline:   
                    updateSweepLine(sweepline, currentEvent, circles)

            # LEFT RECTANGLE EVENT
            if currentEvent["obj"].typeEvent == "LEFTRECTANGLE":
                
                if int(currentEvent["obj"].circle.name) not in sweepline:   
                    updateSweepLine(sweepline, currentEvent, circles)
            
            
            print(f'Type of the event: {currentEvent["obj"].typeEvent}')
            print(f'Check intersection up and below the circle {currentEvent["obj"].circle.name}')
            intersectionPointsList = findIntersection(sweepline, int(currentEvent["obj"].circle.name), circles)
            if (len(intersectionPointsList[0]) != 0) or (len(intersectionPointsList[1]) != 0):
                print("\nThe elements of D do not form a packing of R")
                break
            else:
                print("No intersections found at this iteration\n")

            # RIGHT EVENT
            if currentEvent["obj"].typeEvent == "RIGHT":
                deleteElementFromSweepLine(sweepline, int(currentEvent["obj"].circle.name))
            
            # print sweepline
            if len(sweepline) == 0:
                print("\nThe sweepline is empty!\n")
            else:
                print("Sweepline:")
                pretty(sweepline)
            
            screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation)

        else: # outside the rectangle R
            if currentEvent["obj"].typeEvent == "LEFT":

                for eve in eventsQueue.copy(): # NOTE we could improve it looping only on events after the currentEvent
                    if eventsQueue[eve]["obj"].pointEvent > currentEvent["obj"].pointEvent:
                        # Adding a new intersection rectangle event
                        beforeTmp = eventsQueue[eve]["before"]
                        objName = currentEvent["obj"].circle.name + "RECTANGLE"
                        copyObj = deepcopy(currentEvent["obj"])
                        copyObj.name = objName
                        copyObj.pointEvent = 0
                        copyObj.typeEvent = "LEFTRECTANGLE"

                        eventsQueue[objName] = {
                            "obj" : copyObj,
                            "after" : eve,
                            "before" : beforeTmp
                        }

                        eventsQueue[eve]["before"] = objName
                        eventsQueue[beforeTmp]["after"] = objName
                        break
                if int(currentEvent["obj"].circle.name) not in sweepline: 
                    updateSweepLine(sweepline, currentEvent, circles)

        count += 1
        if currentEvent["after"] is None:
            print("The event queue is empty!\n")
            print("The elements of D form a packing of R")
            break
        currentEvent = eventsQueue[currentEvent["after"]] # access to the next event

def isPointInCircle(circle, p, scalefactor = 50):
    cx = circle.point.x * scalefactor
    cy = circle.point.y * scalefactor
    d = math.sqrt( (p.x - cx) ** 2  + (p.y - cy) ** 2)

    print(f"\ncircle: {circle}, circleScaled: {cx} {cy}, p: {p}")
    print(f"d: {d}, round(d): {round(d)}")
    print(f"circle.radius: {circle.radius}, circle.radiusScaled: {circle.radius * scalefactor}\n")

    if int(np.ceil(d)) < circle.radius * scalefactor:
        return True
    else:
        return False

def checkIntersection(sweepline, allIntersectionNotChecked, circles):
    if len(allIntersectionNotChecked) == 0:
        return

    print("\nBefore allIntersectionNotChecked")
    print(sweepline)
    print(allIntersectionNotChecked)

    lenSweep = len(sweepline)
    for pointsInter in allIntersectionNotChecked.copy():
        notIncircle = []
        for point in allIntersectionNotChecked[pointsInter]["points"]:
            count = 0
            for p in sweepline:
                if not isPointInCircle(circles[p], point):
                    count += 1

            if count == lenSweep:
                notIncircle += [point]
                print(notIncircle)

        if len(notIncircle) > 0:
            allIntersectionNotChecked[pointsInter]["points"] = notIncircle
        else:
            del allIntersectionNotChecked[pointsInter]

        #allIntersectionNotChecked[pointsInter]["points"] = notIncircle
        
    print("\nAfter allIntersectionNotChecked")
    print(allIntersectionNotChecked)

def pNotIn(point, pointsToCheck, height, width, scalefactor = 50):

    # not add value out the the rectangle
    if point.x > width.y * scalefactor or point.y > height.y * scalefactor:
        return False

    # Negative values are out of the screen, they are not added in the allIntersectionNotChecked
    if point.x < 0 or point.y < 0:
        return False

    # Duplicate values are not added in allIntersectionNotChecked
    for p in pointsToCheck:
        if point.x == p.x and point.y == p.y:
            return False
    
    return True

def planeSweepCover(x, y, circles, animation):
    events = []
    for circle in circles:
        events.append(Event(circle, circle.point.x - circle.radius, "LEFT"))
        events.append(Event(circle, circle.point.x + circle.radius, "RIGHT"))

    print("\nNot ordered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")

    # A way that can be fastest, especially if your list has a lot 
    # of records, is to use operator.attrgetter("count")
    try: import operator
    except ImportError: keyfun= lambda x: x.count # use a lambda if no operator module
    else: keyfun= operator.attrgetter("pointEvent") # use operator since it's faster than lambda

    events.sort(key=keyfun, reverse=False) # sort in-place

    print("\nOrdered events:")  
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")
    print("\n") 

    '''
    # example of eventsQueue
    eventsQueue = {
        "event1" : {
            "obj" : event1,
            "before" : "event0",
            "after" : "event2",
        },

        "event2" :{
            "obj" : event2,
            "before" : "event1",
            "after" : "event3",
        }
    }

    # example of sweepline
    sweepline = {
        0 : {               # name of this circle
            "up" : None,    # name of upper circle
            "me" : 0,       # name of this circle
            "below" : 2,    # name of lower circle
        },

        2 :{
            "up" : 0,
            "me" : 2,
            "below" : 1,
        }

        1 :{
            "up" : 2,
            "me" : 1,
            "below" : None,
        }
    }
    '''

    sweepline = {}

    # We set the priority through a dictionary of events
    eventsQueue = {}
    eventsQueue[events[0].name] = {"obj": events[0], "before" : None, "after" : events[1].name}

    for i in range(1, len(events[1::])):
        eventsQueue[events[i].name] = {"obj": events[i], "before" : events[i-1].name, "after" : events[i+1].name}

    eventsQueue[events[-1].name] = {"obj": events[-1], "before" : events[-2].name, "after" : None}

    isThereAnIntersection = False
    count = 0
    currentEvent = eventsQueue[events[0].name]
    allIntersectionNotChecked = {}
    # Swap line among the events
    while True:
        print(f"Iteration: {count}")
        print(f'Name current obj: {currentEvent["obj"].name}')
        print(eventsQueue)
        # pretty(eventsQueue) # pretty view of data
        print(f"Length eventsQueue: {len(eventsQueue)} \n")

        # inside the rectangle R
        if x.x <= currentEvent["obj"].pointEvent <= x.y: 

            # LEFT EVENT
            if currentEvent["obj"].typeEvent == "LEFT":
                
                if int(currentEvent["obj"].circle.name) not in sweepline:   
                    updateSweepLine(sweepline, currentEvent, circles)

            # LEFT RECTANGLE EVENT
            if currentEvent["obj"].typeEvent == "LEFTRECTANGLE":
                
                if int(currentEvent["obj"].circle.name) not in sweepline:   
                    updateSweepLine(sweepline, currentEvent, circles)
            
            
            print(f'Type of the event: {currentEvent["obj"].typeEvent}')
            print(f'Check intersection up and below the circle {currentEvent["obj"].circle.name}')
            intersectionPointsList = findIntersection(sweepline, int(currentEvent["obj"].circle.name), circles)

            if (len(intersectionPointsList[0]) != 0) or (len(intersectionPointsList[1]) != 0):
                isThereAnIntersection = True
                index_circle = int(currentEvent["obj"].circle.name)

                if index_circle not in allIntersectionNotChecked: # if not exists we create it
                    allIntersectionNotChecked[index_circle] = {"points": []}

                pointsToCheck = allIntersectionNotChecked[index_circle]["points"]

                if len(intersectionPointsList[0]) == 1:
                    if pNotIn(intersectionPointsList[0][0], pointsToCheck, x, y):
                        allIntersectionNotChecked[index_circle]["points"] += [ intersectionPointsList[0][0] ]
                elif len(intersectionPointsList[0]) == 2:
                    firstP = intersectionPointsList[0]
                    if firstP[0].x == firstP[1].x and firstP[0].y == firstP[1].y:
                        if pNotIn(firstP[0], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ firstP[0] ]
                    else:
                        if pNotIn(firstP[0], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ firstP[0] ]
                        if pNotIn(firstP[1], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ firstP[1] ]
                if len(intersectionPointsList[1]) == 1:
                    if pNotIn(intersectionPointsList[1][0], pointsToCheck, x, y):
                        allIntersectionNotChecked[index_circle]["points"] += [ intersectionPointsList[1][0] ]
                elif len(intersectionPointsList[1]) == 2:
                    thirdP = intersectionPointsList[1]
                    if thirdP[0].x == thirdP[1].x and thirdP[0].y == thirdP[1].y:
                        if pNotIn(thirdP[0], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ thirdP[0] ]
                    else:
                        if pNotIn(thirdP[0], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ thirdP[0] ]
                        if pNotIn(thirdP[1], pointsToCheck, x, y):
                            allIntersectionNotChecked[index_circle]["points"] += [ thirdP[1] ]
            else:
                print("No intersections found at this iteration\n")

            checkIntersection(sweepline, allIntersectionNotChecked, circles)
            
            # RIGHT EVENT
            if currentEvent["obj"].typeEvent == "RIGHT":
                deleteElementFromSweepLine(sweepline, int(currentEvent["obj"].circle.name))
                

            # print sweepline
            if len(sweepline) == 0:
                print("\nThe sweepline is empty!\n")
            else:
                print("Sweepline:")
                pretty(sweepline)
            
            screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation)

        else: # outside the rectangle R
            if currentEvent["obj"].typeEvent == "LEFT":

                for eve in eventsQueue.copy(): # NOTE we could improve it looping only on events after the currentEvent
                    if eventsQueue[eve]["obj"].pointEvent > currentEvent["obj"].pointEvent:
                        # Adding a new intersection rectangle event
                        beforeTmp = eventsQueue[eve]["before"]
                        objName = currentEvent["obj"].circle.name + "RECTANGLE"
                        copyObj = deepcopy(currentEvent["obj"])
                        copyObj.name = objName
                        copyObj.pointEvent = 0
                        copyObj.typeEvent = "LEFTRECTANGLE"

                        eventsQueue[objName] = {
                            "obj" : copyObj,
                            "after" : eve,
                            "before" : beforeTmp
                        }

                        eventsQueue[eve]["before"] = objName
                        eventsQueue[beforeTmp]["after"] = objName
                        break
                if int(currentEvent["obj"].circle.name) not in sweepline: 
                    updateSweepLine(sweepline, currentEvent, circles)

        count += 1
        if currentEvent["after"] is None:
            print("The event queue is empty!\n")

            if len(allIntersectionNotChecked)>0:
                print("The elements of D do not form a cover of R\nThese points do not belong within any circle:")
            else:
                if isThereAnIntersection:
                    print("The elements of D form a cover of R")
                else:
                    print("The elements of D do not form a cover of R")
            print(allIntersectionNotChecked)
            screenShotPlaneSweep(x, y, circles, currentEvent["obj"], intersection=allIntersectionNotChecked, animation=animation)
            break
        currentEvent = eventsQueue[currentEvent["after"]] # access to the next event

def main():
    # Read different examples of input
    file1 = open('ExampleInput', 'r')
    #file1 = open('ExampleInputTest', 'r')
    #file1 = open('ExampleInputTest2', 'r')
    #file1 = open('ExampleInputTest3', 'r')

    # Parse The input
    x, y ,circles = parseInput(file1)

    # This is only to see what happens, not plane sweep algorithm applied here
    draw(x,y,circles) 

    start = time.time()
    planeSweepPacking(x, y, circles, animation = False)
    #planeSweepCover(x, y, circles, animation = False)
    end1 = time.time() - start

    print(f"\nTime for plane sweep: {end1}")

if __name__ == "__main__":
    main()