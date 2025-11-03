cd hex-game/board-generator/

#Compile
gcc -DBOARD_DIM=3 -o hex hex.c

#Run
./hex

#For Many-Boards-In-One command
chmod +x build_and_run.sh
./build_and_run.sh 3 4 5 6 7 8 9 10 11
