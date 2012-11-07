from pylab import *

data = loadtxt('test_results', delimiter=',');
i_max = 0; t_max = 1; num_threads = 2; time= 3; normalized = 4;

hold(True)

# plot the number of threads against the time for each variation of i_max
value = data[:, i_max] == 1000;
plot( data[value, num_threads], data[value, time])
xlabel('number of threads')
ylabel('time (seconds)')
#label('i_max = 1000')

value = data[:, i_max] == 10000;
plot( data[value, num_threads], data[value, time])
xlabel('number of threads')
ylabel('time (seconds)')
#label('i_max = 10000')

value = data[:, i_max] == 100000;
plot( data[value, num_threads], data[value, time])
xlabel('number of threads')
ylabel('time (seconds)')
#label('i_max = 100000')

value = data[:, i_max] == 1000000;
plot( data[value, num_threads], data[value, time])
xlabel('number of threads')
ylabel('time (seconds)')
#label('i_max = 1000000')

value = data[:, i_max] == 10000000;
plot( data[value, num_threads], data[value, time])
xlabel('number of threads')
ylabel('time (seconds)')
#label('i_max = 10000000')

legend(("i_max = 1000", "i_max = 10000", "i_max = 100000", "i_max = 1000000",
"i_max = 10000000"))

show()
