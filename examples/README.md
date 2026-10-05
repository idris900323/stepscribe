# Example STEP files

No third-party CAD files are committed to this repository (licenses vary). To try `stepscribe` on
real robots, download a few yourself and keep them in a folder that is not tracked by git, for
example `robot_steps/` (already in `.gitignore`).

Open-source robot projects that publish STEP files (check each repository's license before
redistributing anything):

| Project | What it is | License (as reported by GitHub) |
|---|---|---|
| [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | small 5-DOF servo arm, part and assembly STEP files | Apache-2.0 |
| [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) | 6-axis desktop arm, many medium parts and a full assembly | CERN-OHL-W-2.0 |
| [AngelLM/Thor](https://github.com/AngelLM/Thor) | printable 6-axis arm | CC-BY-SA-4.0 |
| [jess-moss/koch-v1-1](https://github.com/jess-moss/koch-v1-1) | leader/follower arm (the assembly references other STEP files) | Apache-2.0 |
| [SkyentificGit/SmallRobotArm](https://github.com/SkyentificGit/SmallRobotArm) | small arm axes | GPL-3.0 |
| [Source-Robotics/PAROL6-Desktop-robot-arm](https://github.com/Source-Robotics/PAROL6-Desktop-robot-arm) | desktop arm (two small STEP plates in the repo) | GPL-3.0 |
| [menloresearch/asimov-v0](https://github.com/menloresearch/asimov-v0) | humanoid lower body, one very large assembly (Git LFS, about 100 MB) | CERN-OHL-S-2.0 |
| [nasa-jpl/open-source-rover](https://github.com/nasa-jpl/open-source-rover) | rover; only board and connector STEP models are in the repo | Apache-2.0 |

Other good sources: public FRC team CAD releases, GrabCAD models, and your own designs.

Some assemblies only *reference* part files stored next to them. If `stepscribe` reports "assembly
structure but no geometry", download the referenced files into the same folder or export a
self-contained STEP.

Try a folder run and look at timings and failures:

```bash
python scripts/batch_eval.py robot_steps/ --jobs 2 --csv out/eval.csv
stepscribe pack robot_steps/ -o out --jobs 2
```
