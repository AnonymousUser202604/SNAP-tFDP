#!/usr/bin/env bash
# transform dot files
(echo "graph G {"; tail -n +2 ./data/APH.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/APH.dot
(echo "graph G {"; tail -n +2 ./data/aircraft.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/aircraft.dot
(echo "graph G {"; tail -n +2 ./data/co_author_8391.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/co_author_8391.dot
(echo "graph G {"; tail -n +2 ./data/socfb-Yale4.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/socfb-Yale4.dot
(echo "graph G {"; tail -n +2 ./data/ACO.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/ACO.dot
(echo "graph G {"; tail -n +2 ./data/socfb-UF21.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/socfb-UF21.dot
(echo "graph G {"; tail -n +2 ./data/soc-Flickr-ASU.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/soc-Flickr-ASU.dot
(echo "graph G {"; tail -n +2 ./data/com-dblp.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/com-dblp.dot
(echo "graph G {"; tail -n +2 ./data/com-amazon.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/com-amazon.dot
(echo "graph G {"; tail -n +2 ./data/com-youtube.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/com-youtube.dot
(echo "graph G {"; tail -n +2 ./data/com-orkut.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/com-orkut.dot
(echo "graph G {"; tail -n +2 ./data/com-lj.txt | awk '{print $1 " -- " $2}'; echo "}") > ./data/com-lj.dot

# transform mtx files
cmake -S . -B build
cmake --build build --target txt2mtx
./tools/txt2mtx
