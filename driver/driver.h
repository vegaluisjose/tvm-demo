#ifndef Driver_H_
#define Driver_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* DriverHandle;

/* allocate Driver */
DriverHandle DriverAlloc();

/* deallocate Driver */
void DriverDealloc(DriverHandle handle);

/* read Driver register or memory */
int DriverRead(DriverHandle handle, int id, int addr);

/* write Driver register or memory */
void DriverWrite(DriverHandle handle, int id, int addr, int value);

/* reset Driver for n clock cycles */
void DriverReset(DriverHandle handle, int n);

/* run Driver for n clock cycles */
void DriverRun(DriverHandle handle, int n);


#ifdef __cplusplus
}
#endif

#endif  // Driver_H_
