# Manhattan Provisioning
  This document lists steps for Manhattan provision and some issues
  
  
  ## What is in the usb drive
  Two files need to appear in the usb drive,
  1. a zipped or tar containing greengrass config file, certficates
  2. Network.json
  
  ###  zipped config file
  Downloaded when we add the new device.
  
  ### Network.json
  The network json file conatines the information of wifi, the format is as
  follows:
  
  Wifi:
      SSID ""
      Passcode ""
  
  
  Some issues which may be improved in the future
  1. The device cannot detect the usb drive sometimes since the usb file is hard
  coded as sda1
  2. The device is sometimes connected to ethernet (without password required), in this case
  Network.json may not be required
  ~                                                                                                                                                                        
  ~                                                                                                                                                                        
  ~                                                                                                                                                                        
  ~                                                
