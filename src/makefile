all: animation.gif

%: %.c
	gcc -O2 -o $@ $< -lm

results/0.csv: seq
	./seq

animation.gif: results/0.csv makegif.py
	python3 makegif.py

clean:
	rm -f seq
	rm -f results/*