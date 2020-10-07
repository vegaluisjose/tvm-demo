#ifndef DEVICE_H_
#define DEVICE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* DeviceHandle;

/* allocate device */
DeviceHandle DeviceAlloc();

/* deallocate device */
void DeviceDealloc(DeviceHandle handle);

/* read device register or memory */
int DeviceRead(DeviceHandle handle, int id, int addr);

/* write device register or memory */
void DeviceWrite(DeviceHandle handle, int id, int addr, int value);

/* reset device for n clock cycles */
void DeviceReset(DeviceHandle handle, int n);

/* run device for n clock cycles */
void DeviceRun(DeviceHandle handle, int n);


#ifdef __cplusplus
}
#endif

#endif  // DEVICE_H_
