#!/bin/bash

# sentence="He said last week 's tsunami and the massive underwater earthquake that triggered it has affected millions in Asia and Africa ."
#sentence="Warner told supporters outside of the University of Virginia Friday that he will not seek a sixth term in the 2008 elections ."
sentence="With the Tokyo Olympics less than a year away, speculation is growing in Japan over who will be the final torchbearer to light the cauldron in the new Olympic Stadium, in the traditional ritual that begins every Games. Some predict a famous Japanese athlete, such as retired baseball player Ichiro Suzuki, will do the honors. Others say it will be an ordinary, but symbolically important, person. Picked to light the flame in 1964, the last time Tokyo hosted the Games, was Yoshinori Sakai, a 19-year-old college athlete born in Hiroshima on Aug. 6, 1945, the day of the U.S. atomic bombing - a choice meant to highlight Japan's remarkable post-war reconstruction."

curl localhost:5050/tag_sentence -d '{"sentence": "'"${sentence}"'"}' -H 'Content-Type: application/json'