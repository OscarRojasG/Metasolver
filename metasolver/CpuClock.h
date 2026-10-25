/*
 * CpuClock.h
 *
 * Reloj compatible con std::chrono que mide CPU time del proceso
 * (user + system, sumando todos sus hilos).
 */

#ifndef CPUCLOCK_H_
#define CPUCLOCK_H_

#include <chrono>
#include <time.h>

namespace metasolver {

struct cpu_clock {
	using duration = std::chrono::nanoseconds;
	using rep = duration::rep;
	using period = duration::period;
	using time_point = std::chrono::time_point<cpu_clock, duration>;
	static constexpr bool is_steady = true;

	static time_point now() noexcept {
		timespec ts;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
		return time_point(std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec));
	}
};

}

#endif /* CPUCLOCK_H_ */
