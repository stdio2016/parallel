#define RED 0
#define GREEN 256
#define BLUE 512
#define HIST_SIZE 768
kernel void histogram(
	global unsigned char *str,
	global int *range,
	global unsigned int *result)
{
	local unsigned int count[HIST_SIZE];
	for (int i = 0; i < HIST_SIZE; i++) count[i] = 0;

	// get my work range
	int id = get_global_id(0);
	int thread_count = get_global_size(0);
	int start = range[id];
	int end = range[id+1];

	// parse number in GPU!
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

	for (int i = 0; i < HIST_SIZE; i++) {
		result[id * HIST_SIZE + i] = count[i];
	}
}
