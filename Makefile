CC:=cc
CFLAGS:= -Wall -Wextra -O0 -fsanitize=undefined,address -g
LDFLAGS:= $(CFLAGS) -lm -lgsl -lgslcblas -g


.PHONY: unzip zip clean
.DELETE_ON_ERROR: test 

all: test mnist

nn.c: gan.o

mnist: mnist.o nn.o gan.o
	$(CC) $(LDFLAGS) $^ -o $@

test: test.o nn.o gan.o
	$(CC) $(LDFLAGS) $^ -o $@
	./test
	-rm test
unzip: 
	unzip data.zip
	rm data.zip

zip:
	zip data.zip mnist_*
	rm mnist_*

clean:
	-rm *.o mnist 



	
