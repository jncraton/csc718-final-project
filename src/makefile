all: media/omp.gif media/cuda.gif media/seq.gif

compare: media/omp.gif media/seq.gif media/cuda.gif
	cmp $< media/omp.gif
	cmp $< media/cuda.gif

dist/cuda: cuda.cu
	nvcc --compiler-options -O3 $< -o $@ --compiler-options -lm --compiler-options -fopenmp -arch=sm_60

dist/%: %.c shared.h main.h
	g++ -Wall -Wextra -pg -march=ivybridge -O3 -o $@ $< -lm -fopenmp

media/%.gif: dist/% render.py
	rm -rf $<-results
	mkdir $<-results
	cd $<-results && ../../$<
	python3 render.py "$@" > /dev/null 2>&1 
	gifsicle -O3 --batch "$@" > /dev/null 2>&1

clean:
	rm -rf dist/*
	touch dist/.gitignore
	rm -f media/*.gif
	rm -f *.gif