all: animation.gif

%: %.c
	gcc -ffast-math -O3 -o $@ $< -lm -fopenmp 

results/0.csv: sim-omp
	./sim-omp

animation.gif: results/0.csv makegif.py
	python3 makegif.py > /dev/null 2>&1 
	gifsicle -O3 --batch animation.gif > /dev/null 2>&1

clean:
	rm -f seq
	rm -f sim-*
	rm -f results/*
	rm -f *.gif