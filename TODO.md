2024-07-25
code:
- [x] decompose tasks, for example, approaching, grasping, lifting
- [x] dense reward for approaching
- [x] better DSA for decomposed tasks, for example different representation for different subtasks


exp:
- [x] debug orginal GAS baseline, try if the code is broken
- [x] report the original GAS on grasping standing needle

2024-08-21
- [ ] support stereo cam to get depth
- [ ] support arbitrary cam pose, maybe need to reproject
- [ ] continuous action space
- [ ] support quantized action in continuous action space
- [ ] interpolating the next step desired pose with a smoother method, in cartesian space
- [ ] controller's action need to deal with corner cases, like hitting the boundary of workspace, maybe mapping to another action
- [ ] tune dense reward for approaching.
- [ ] change masking object based on stereo's depth
- [ ] broadcasting dsa scalar signals to channel's level
- [ ] smoother motion by remove the blocking of psm's commands (maybe threading)


2024-08-27 (after trials in robots)
code
- [ ] needle on uneven z-level ground
- [ ] reduce hitting the ground
- [ ] define task that with (1) only needle (2) limitted grasp trials (3) endoscopic depth setting, cam distance, angle, and the choices of masks (4) limited timesteps, wwwww