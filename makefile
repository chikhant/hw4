all: clean
	g++ -O3 -o mm mm.cc

clean:
	rm -f mm

node:
	qrsh -q eecs117

test: all
	./mm 1024 1024 1024 64
