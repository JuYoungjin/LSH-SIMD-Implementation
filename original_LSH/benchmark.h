#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#define TIMER_SAMPLE_CNT (1000)      /* counter used in calibrating the overhead of measuring time */

unsigned int HiResTime(void);        /* return the current value of time stamp counter */
unsigned int calibrate();
float get_cpb(unsigned int cycle, unsigned int data_len);

#endif
