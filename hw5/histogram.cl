#define RED 0
#define GREEN 256
#define BLUE 512
#define HIST_SIZE 768
kernel void histogram(
	global unsigned char *str,
	global int *range,
	global unsigned int *result)
{
	// private area for frequency distribution
	// to decrease memory bandwidth usage
	// using __private instead of __local will slow down the kernel. I don't know why
	local unsigned int count[HIST_SIZE];

	// cannot initialize a __local array with {0}, so initialize here
	for (int i = 0; i < HIST_SIZE; i++) count[i] = 0;

	// get my work range
	int id = get_global_id(0);
	int thread_count = get_global_size(0);
	int start = range[id];
	int end = range[id+1];

	// parse number in GPU!
	// actually it can only parse /(\d{1,3} \d{1,3} \d{1,3}\n)+/
	// meaning that:
	//   1. no leading spaces, no trailing spaces.
	//   2. there should be exactly 3 numbers in each line.
	//   3. in each line, numbers are separated by exactly 1 space.
	//   4. every number must be between 0 and 255 and have at most 3 digits.
	//   5. numbers are unsigned integer.
	for (int i = start; i < end; ) {
		// red part
		int num, ch;
		num = str[i]-'0';
		ch = str[i+1];
		if (ch != ' ') {
			num = num*10 + ch-'0';
			ch = str[i+2];
			if (ch != ' ') {
				num = num*10 + ch-'0';
				i += 4;
			}
			else i += 3;
		}
		else i += 2;
		++count[RED+num];

		// green part
		num = str[i]-'0';
		ch = str[i+1];
		if (ch != ' ') {
			num = num*10 + ch-'0';
			ch = str[i+2];
			if (ch != ' ') {
				num = num*10 + ch-'0';
				i += 4;
			}
			else i += 3;
		}
		else i += 2;
		++count[GREEN+num];

		// blue part
		num = str[i]-'0';
		ch = str[i+1];
		if (ch != '\n') {
			num = num*10 + ch-'0';
			ch = str[i+2];
			if (ch != '\n') {
				num = num*10 + ch-'0';
				i += 4;
			}
			else i += 3;
		}
		else i += 2;
		++count[BLUE+num];
	}

	// write result to global memory
	for (int i = 0; i < HIST_SIZE; i++) {
		result[id * HIST_SIZE + i] = count[i];
	}
}
