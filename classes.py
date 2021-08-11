import numpy as np
from time import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from json import loads
from bisect import bisect_left
from hashids import Hashids
import math
from threading import Lock

from sensor_msgs import msg

# Utils
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def parsed(msg):
    people = [
        Data(p.body.landmarks[5:7], p.body.location.x, p.id)
        for p in msg.people
    ]
    people.sort(key=lambda s: s.depth, reverse=True)
    for i in range(len(people) - 1):
        people[i].adjust(people[i + 1:])
    return people

def compare(pat1, pat2, method):
    s1 = pat1.med_sat()
    s2 = pat2.med_sat()
    gray = 22
    if ((s1 <= gray or s2 <= gray) and abs(s1 - s2) <= 5) or (s1 <= gray
                                                              and s2 <= gray):
        return cv2.compareHist(pat1.val(), pat2.val(), method)
    elif s1 > gray and s2 > gray and abs(s1 - s2) <= 10:
        return cv2.compareHist(np.append(pat1.hue(), pat1.sat()),
                               np.append(pat2.hue(), pat1.sat()),
                               method)
    else:
        return 0


class Pattern:
    def __init__(self):
        self.super_hue = np.zeros(180, dtype=np.float32)
        self.super_sat = np.zeros(256, dtype=np.float32)
        self.super_val = np.zeros(256, dtype=np.float32)
        self.size = 0
        self.est_sat = None

    def hue(self):
        return cv2.normalize(self.super_hue, self.super_hue)

    def sat(self):
        return cv2.normalize(self.super_sat, self.super_sat)

    def val(self):
        return cv2.normalize(self.super_val, self.super_val)

    def med_sat(self):
        if self.est_sat == None:
            self.est_sat = (np.abs(np.cumsum(self.sat()) - 0.5)).argmin()
        return (np.abs(np.cumsum(self.sat()) - 0.5)).argmin()

    def update(self, frame):
        if frame.shape[0] * frame.shape[1] > 80:
            self.super_hue = np.add(
                self.super_hue,
                np.bincount(frame[:, :, 0].flatten(),
                            minlength=180).astype('float32'))
            self.super_sat = np.add(
                self.super_sat,
                np.bincount(frame[:, :, 1].flatten(),
                            minlength=256).astype('float32'))
            self.super_val = np.add(
                self.super_val,
                np.bincount(frame[:, :, 2].flatten(),
                            minlength=256).astype('float32'))
            self.size += 1


hashids = Hashids()


class Person:
    def __init__(self, id):
        self.id = id
        self.pattern = Pattern()
        self.last_seen = time()
        self.counter = 0
        self.name = hashids.encode(int(time()))
        self.ambiguous = False
        self.bd = time()


    def merge(self, other):
        self.name = other.name
        self.pattern.size += other.pattern.size
        self.pattern.super_hue = np.add(self.pattern.super_hue,
                                        other.pattern.super_hue)
        self.pattern.super_sat = np.add(self.pattern.super_sat,
                                        other.pattern.super_sat)
        self.pattern.super_val = np.add(self.pattern.super_val,
                                        other.pattern.super_val)

    def ctype(self):
        return self.pattern.med_sat()>22

    def __repr__(self):
        return str(self.name)


class Data:
    def __init__(self, shoulders, depth, id):
        self.x0, self.x1 = tuple(sorted([int(s.x) for s in shoulders]))
        self.y0, self.y1 = tuple(sorted([int(s.y) for s in shoulders]))
        self.depth = depth
        self.adj = int(
            (((self.x0 - self.x1)**2 + (self.y0 - self.y1)**2)**0.5) / 10)
        self.y0 -= self.adj
        self.y1 += self.adj
        self.id = id
        self.bounds()

    def adjust(self, shoulders):
        for s in shoulders:
            if s.x0 < self.x0 < s.x1:
                self.x0 = int(s.x1 + s.adj)
            if s.x0 < self.x1 < s.x1:
                self.x1 = int(s.x0 - s.adj)
            self.bounds()

    def bounds(self):
        if self.y0 < 0:
            self.y0 = 0
        if self.x0 < 0:
            self.x0 = 0
        if self.y1 > 479:
            self.y1 = 479
        if self.x1 > 639:
            self.x1 = 639

    def __repr__(self):
        return str(self.x0) + ', ' + str(self.y0) + ', ' + str(
            self.x1) + ', ' + str(self.y1)


class Pattern_Match:
    def __init__(self, thresholds=(0.50,0.30), req_frames=10, frame_rate=3, method=(cv2.HISTCMP_CORREL, lambda s, t: s >= t, False), mem_length = 600, draw=False):
        self.memory = []
        self.current = []
        self.recent_frames = []
        self.recent_times = []
        self.frame_count = -1

        self.colort, self.grayt = thresholds
        self.req_frames = req_frames
        self.mem_req = req_frames
        self.frame_rate = frame_rate
        self.method, self.comp, self.sigmoid = method

        self.merges = 0

        self.draw = draw

        self.check = [False, False]
        self.lock = Lock()
        self.people_msgs = []
        self.mem_length = mem_length


    def image_callback(self, msg):
        #Check image messages are working
        if not self.check[0]:
            print('IMG callback success')
            self.check[0] = True
        t = msg.header.stamp
        try:
            cv2_img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            raise ValueError('Image not bgr8 format.')
        else:
            with self.lock:
                if len(self.recent_frames) >= 300:
                    self.recent_frames.pop(0)
                    self.recent_times.pop(0)
                self.recent_frames.append(cv2_img)
                self.recent_times.append(t.secs + t.nsecs * 10**-9)

    def people_callback(self, people_msg):
        #buffer
        self.people_msgs.append(people_msg)
        if len(self.people_msgs) < 120:
            return
        msg = self.people_msgs.pop(0)

        #Check that people messages is working
        if not self.check[1]:
            print('People callback success')
            self.check[1] = True
        self.frame_count += 1

        if msg.people and self.recent_frames:
            #prune memory
            self.memory = [
                person for person in self.memory
                if time() - person.last_seen <= self.mem_length
            ]
            data = parsed(msg)
            #get and process image
            t = msg.people[0].body.header.stamp
            img = self.closest_frame(t.secs + t.nsecs * 10**-9)
            if type(img) == np.ndarray:
                hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)

                #update memory
                current = [d.id for d in data]
                recent_ids = [p.id for p in self.current]
                recently_left = list(set(recent_ids) - set(current))
                for person in self.current[:]:
                    if person.id in recently_left:
                        if person.pattern.size >= self.mem_req:
                            self.memory.append(person)
                        self.current.remove(person)
                        recent_ids.remove(person.id)
                for d in data:
                    cut = hsv[d.y0:d.y1, d.x0:d.x1]
                    if d.id in recent_ids:
                        person = self.current[recent_ids.index(d.id)]
                        # Handle boundary switch issue (shoulder tracking errors)
                        if person.ambiguous:
                            test = Person(d.id)
                            test.pattern.update(cut)
                            temp_thresh = self.colort if person.ctype() else self.grayt
                            if self.comp(compare(test.pattern, person.pattern,
                                    self.method), temp_thresh):
                                if person.counter % self.frame_rate == self.frame_rate-1:
                                    person.pattern.update(cut)
                                person.counter += 1
                                person.last_seen = time()
                                person.ambiguous = d.x0 == 0 or d.x1 == 639 or d.y0 == 0 or d.y1 == 479
                            else:
                                if person.pattern.size >= self.mem_req:
                                    self.memory.append(person)
                                self.current.remove(person)
                                test.ambiguous = d.x0 == 0 or d.x1 == 639 or d.y0 == 0 or d.y1 == 479
                                self.current.append(test)
                        # Normal within frame
                        else:
                            if person.counter % self.frame_rate == self.frame_rate-1:
                                person.pattern.update(cut)
                            person.counter += 1
                            person.last_seen = time()
                            person.ambiguous = d.x0 == 0 or d.x1 == 639 or d.y0 == 0 or d.y1 == 479
                    # New tracked ID
                    else:
                        new = Person(d.id)
                        new.pattern.update(cut)
                        new.ambiguous = d.x0 == 0 or d.x1 == 639 or d.y0 == 0 or d.y1 == 479
                        self.current.append(new)
                        print(new.name)
                mem_copy = [p for p in self.memory if p.pattern]
                # Look for matches and merge person objects
                for person in self.current:
                    if person.pattern.size >= self.req_frames and mem_copy:
                        closest = (None, self.colort) if person.ctype() else (None, self.grayt)
                        for p in mem_copy:
                            val = compare(person.pattern, p.pattern,
                                            self.method)
                            if self.sigmoid:
                                val = sigmoid(val)
                            if self.comp(val, closest[1]):
                                closest = (p, val)
                        if closest[0]:
                            #Merge older person object
                            if person.bd>closest[0].bd:
                                person.merge(closest[0])
                            else:
                                closest[0].merge(person)
                                self.current[self.current.index(person)] = closest[0]
                            self.merges += 1
                            if closest[0] in self.memory:
                                    self.memory.remove(closest[0])
                            
                # Visualisation
                if self.draw:
                    data_ids = [d.id for d in data]
                    for p in self.current:
                        if p.pattern.size>=self.req_frames and p.id in data_ids:
                            d = data[data_ids.index(p.id)]
                            cv2.rectangle(img, (d.x0, d.y0), (d.x1, d.y1), (255, 0, 0),
                                        2)
                            cv2.putText(
                                img,
                                text=str(
                                    p.name
                                ),
                                org=(d.x0, d.y0),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2,
                                color=(0, 0, 255),
                                thickness=5,
                            )
                        if type(self.draw) == str:
                            cv2.imwrite(self.draw + str(self.frame_count) + '.jpeg',
                                        img)
                        else:
                            cv2.imwrite('frame.jpeg', img)
                            
        #If no people detected clear current and move acceptable person objects to memory.
        elif not msg.people:
            for person in self.current[:]:
                if person.pattern.size >= self.mem_req:
                    self.memory.append(person)
                self.current.remove(person)

    def closest_frame(self, time):
        with self.lock:
            closest_time = min(self.recent_times, key=lambda x:abs(x-time))
            diff = abs(closest_time-time)
            idx = self.recent_times.index(closest_time)
            return self.recent_frames[idx].copy() if diff<0.01 else None

    #Might be faster?
    def closest_frame_old(self, time):
        lst = self.recent_times
        pos = bisect_left(lst, time)
        
        best = None
        if pos == 0:
            best = 0
        if pos == len(lst):
            best = -1
        try:
            if lst[pos] - time < time - lst[pos - 1]:
                best = pos
            else:
                best = pos - 1
        except:
            pass
            # raise ValueError('Not receiving frame data. Check image subscriber')
        diff = self.recent_times[best]-time
        return self.recent_frames[best] if diff<0.05 else None
