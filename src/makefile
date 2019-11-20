all: animation.gif

%: %.c
	gcc -ffast-math -O3 -o $@ $< -lm

results/0.csv: seq
	./seq

animation.gif: results/0.csv makegif.py
	python3 makegif.py
	gifsicle -O3 --batch animation.gif

clean:
	rm -f seq
	rm -f results/*
	rm -f *.gif