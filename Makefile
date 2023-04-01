CC:=cc
CFLAGS:= -Wall -Wextra -lm -lgsl -lgslcblas -g -O0 -fsanitize=undefined,address


.PHONY: unzip zip clean
.DELETE_ON_ERROR: test 

nn.c: gan.o

mnist: mnist.o nn.o gan.o
	$(CC) nn.o gan.o -o $@

test: test.o nn.o gan.o
	$(CC) nn.o gan.c -o $@
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



	
