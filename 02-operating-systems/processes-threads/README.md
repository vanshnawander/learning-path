# Processes and Threads

Fundamental OS concepts for concurrent programming.

## Process vs Thread

| Aspect | Process | Thread |
|--------|---------|--------|
| Memory | Separate | Shared |
| Creation | Heavy | Light |
| Communication | IPC needed | Direct |
| Crash isolation | Yes | No |

## Process Creation (Unix)

```c
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        printf("Child PID: %d\n", getpid());
        exit(0);
    } else {
        // Parent process
        wait(NULL);  // Wait for child
        printf("Parent PID: %d\n", getpid());
    }
    return 0;
}
```

## POSIX Threads

```c
#include <pthread.h>

void *thread_func(void *arg) {
    int id = *(int *)arg;
    printf("Thread %d running\n", id);
    return NULL;
}

int main() {
    pthread_t threads[4];
    int ids[4] = {0, 1, 2, 3};
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, thread_func, &ids[i]);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}
```

Compile: `gcc -pthread program.c`

## Thread Synchronization

### Mutex
```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *critical_section(void *arg) {
    pthread_mutex_lock(&mutex);
    // Only one thread at a time
    pthread_mutex_unlock(&mutex);
    return NULL;
}
```

### Condition Variable
```c
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int ready = 0;

// Producer
pthread_mutex_lock(&mutex);
ready = 1;
pthread_cond_signal(&cond);
pthread_mutex_unlock(&mutex);

// Consumer
pthread_mutex_lock(&mutex);
while (!ready) {
    pthread_cond_wait(&cond, &mutex);
}
pthread_mutex_unlock(&mutex);
```

## Context Switching
- Save registers, program counter
- Switch page tables
- Restore new process state
- Overhead: ~1-10 microseconds
