# Park Vision (E26)

**Park Vision** is a desktop app built by a WPI Interactive Qualifying
Project to help the National Park Service understand visitor use in Acadia
National Park, using still images and video already captured by trail
cameras deployed in the park. Rather than manually reviewing thousands of
photos and hours of footage, it gives you a toolkit of independent
analyses you can mix and match as needed: detect and track people,
bicycles, and vehicles; estimate direction of travel and speed; read
license plates to compute dwell time; and compute per-image occupancy
counts, exporting whichever results you've generated to CSV.

<a href="Poster.jpg"><img src="Poster.jpg" alt="Acadia AI project poster" width="100%"></a>

Read the full writeup in [Report.pdf](<Report.pdf>) for
methodology, evaluation, and results.

## Download

Grab the build for your platform from the
[latest release](https://github.com/ElliotScher/Acadia-AI-E26/releases/latest).

### Windows

1. Download `ImageAnalyzerSetup.exe` and run it.
2. Follow the installer; it adds a Start Menu shortcut and an uninstaller.

### macOS (Apple Silicon)

1. Download `acadia-ai-e26-macos-arm64.zip` and unzip it.
2. Move the executable called `main` wherever you'd like (e.g. `/Applications`) and open it.

### Linux

1. Download `acadia-ai-e26-linux-x86_64.zip` and unzip it.
2. Run the extracted binary (`chmod +x main` first if needed):
   ```bash
   ./main
   ```

## Build From Source

Image Analyzer's GUI (`src/ui`) is built on top of a set of standalone
command-line tools (`src/detection`, `src/processing`, `src/utility`) that
can also batch-process video footage end-to-end outside of the GUI (YOLO
detection → entity tracking → speed/direction/dwell-time computation →
summary report).

### Requirements

- Python >= 3.10 (CI targets 3.13)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on `PATH`
  (used as a fallback to OCR the timestamp burned into camera footage,
  when a frame's filename/mtime doesn't already give a reliable one)
- [Git LFS](https://git-lfs.com/): Model weights (`*.pt`) and sample media
  are tracked via LFS (see `.gitattributes`); run `git lfs pull` after
  cloning if large files show up as pointer text

### Setup

```bash
git clone https://github.com/ElliotScher/Acadia-AI-E26.git
cd Acadia-AI-E26
uv sync
```

`uv sync` installs all dependencies declared in `pyproject.toml`, including
PyTorch, Ultralytics YOLO, OpenCV, PySide6, and SQLAlchemy.

Model weights live under `models/`:

| Path | Purpose |
| --- | --- |
| `models/generic/yolo26s.pt`, `yolov8s.pt` | General-purpose YOLO object detection (people, bikes, vehicles) |
| `models/generic/yolo26s-pose.pt` | Pose keypoints, used for pedestrian direction |
| `models/license_plate/license-plate.pt` | License plate localization |
| `models/vehicle_direction/last.pt` | Vehicle-pose model (fine-tuned on CarFusion) used for vehicle direction; see `src/utility/fetch_vehicle_pose_weights.py` if it needs to be re-downloaded |

### Running from source

```bash
uv run python src/ui/main.py
```

### Command-line pipeline

Each stage below is also runnable independently, and each script's own
`--help` documents its full option set:

1. `src/detection/video_yolo.py` / `image_yolo.py`: Run YOLO over raw video
   frames or images, producing a JSON detection report
2. `src/processing/video_entityprofiler.py`: Track unique entities across
   a video's frames from that report, export the best frame per entity, and
   compute relative/absolute speed and direction
3. `src/processing/report_recalibrator.py`: Recompute absolute speed on an
   existing entity-profiler report against a different reference entity,
   without reprocessing video
4. `src/utility/report_summarizer.py`: Turn an entity-profiler (or
   recalibrator) report into summary statistics: entity counts by type and
   direction, per-video breakdowns, speed statistics
5. `src/processing/video_plateextractor.py` → `plate_dwellprofiler.py`:
   Detect and OCR license plates across video frames, then match repeated
   plate readings to compute vehicle dwell time
6. `src/processing/image_occupancyprofiler.py`: Re-identify and track
   entities chronologically across a folder of images to compute occupancy
   counts

Utility scripts under `src/utility/` also include benchmarks
(`vehicle_speed_benchmark.py`, `vehicle_direction_benchmark.py`,
`pedestrian_direction_benchmark.py`, `plate_ocr_benchmark.py`,
`speed_distribution_comparator.py`) that validate these algorithms against
labeled/ground-truth datasets, and `scraper.py` for bulk-downloading photos
from a Spypoint camera account.

### Testing

```bash
uv run pytest
```

Tests live under `tests/`, mirroring the `src/` package layout, and run
against sample media in `tests/data/`.

### Building an executable

Desktop builds are produced with PyInstaller (`main.spec`), driven by
`.github/workflows/build.yml` for Linux, Windows, and macOS:

```bash
uv run pyinstaller main.spec
```

On Windows, the resulting binary is wrapped into an installer with
[Inno Setup](https://jrsoftware.org/isinfo.php) via `installer/main.iss`.
## License

Licensed under the [GNU AGPLv3](LICENSE).

<details>
<summary><h2 style="display: inline;">Bibliography</h2></summary>

<pre>
<i>Acadia National Park final transportation plan / Environmental impact statement</i>. (2019,
    March). National Park Service.
    <a href="https://parkplanning.nps.gov/document.cfm?parkID=203&amp;documentID=94071">https://parkplanning.nps.gov/document.cfm?parkID=203&amp;documentID=94071</a>

Ahad, A., Kidwai, F. A., &amp; Alqadhi, S. (2026). AI-augmented real-time parking occupancy
    detection and prediction: Integrating computer vision with user behavioral insights
    for Delhi’s mixed-traffic ecosystem. <i>Transportation</i>.
    <a href="https://doi.org/10.1007/s11116-026-10767-1">https://doi.org/10.1007/s11116-026-10767-1</a>

Albers, J. L., Wildhaber, M. L., Green, N. S., Struckhoff, M. A., &amp; Hooper, M. J.
    (2023). Visitor use and activities detected using trail cameras at forest
    restoration sites. <i>Ecological Restoration</i>, <i>41</i>(4), 199–212.
    <a href="https://doi.org/10.3368/er.41.4.199">https://doi.org/10.3368/er.41.4.199</a>

Arshad, J., Ijaz, C. A., Yousaf, A., Habib, S., Rehman, A. U., Abid, H., &amp; Asif, R. M.
    (2022). Implementation of an intelligent parking system. <i>2022 International
    Conference on Engineering and Emerging Technologies (ICEET)</i>, 1–11.
    <a href="https://doi.org/10.1109/ICEET56468.2022.10007136">https://doi.org/10.1109/ICEET56468.2022.10007136</a>

Bai, L., Wu, C., Xie, F., &amp; Wang, Y. (2021). Crowd density detection method based on
    crowd gathering mode and multi-column convolutional neural network. <i>Image and Vision
    Computing</i>, <i>105</i>, 104084. <a href="https://doi.org/10.1016/j.imavis.2020.104084">https://doi.org/10.1016/j.imavis.2020.104084</a>

Bai, Y., Zou, Q., Chen, X., Li, L., Ding, Z., &amp; Chen, L. (2023). Extreme low-resolution
    action recognition with confident spatial-temporal attention transfer. <i>International
    Journal of Computer Vision</i>, <i>131</i>(6), 1550–1565.
    <a href="https://doi.org/10.1007/s11263-023-01771-4">https://doi.org/10.1007/s11263-023-01771-4</a>

Butoto, J., Liu, X., &amp; Sando, T. (2026). Dwell time estimation using periodic image
    captures and deep learning. <i>The International FLAIRS Conference Proceedings</i>, <i>39</i>(1).
    <a href="https://doi.org/10.32473/flairs.39.1.141550">https://doi.org/10.32473/flairs.39.1.141550</a>

Choi, W., Chao, Y.-W., Pantofaru, C., &amp; Savarese, S. (2014). Discovering Groups of
    People in Images. In D. Fleet, T. Pajdla, B. Schiele, &amp; T. Tuytelaars (Eds.),
    <i>Computer Vision – ECCV 2014</i> (Vol. 8692, pp. 417–433). Springer International
    Publishing. <a href="https://doi.org/10.1007/978-3-319-10593-2_28">https://doi.org/10.1007/978-3-319-10593-2_28</a>

Eco-Counter. (n.d.). <i>Counting solutions to understand user flows</i>. Retrieved April 30,
    2026, from <a href="https://www.eco-counter.com/solutions/counting-solutions">https://www.eco-counter.com/solutions/counting-solutions</a>

Elek, O., Thomas, M. M., &amp; Forbes, A. (2019). <i>Learning patterns in sample distributions
    for Monte Carlo variance reduction</i> (arXiv:1906.00124; Version 1). arXiv.
    <a href="https://doi.org/10.48550/arXiv.1906.00124">https://doi.org/10.48550/arXiv.1906.00124</a>

Fleming, D. (2016). As popular Acadia turns 100, there’s no room at the top. <i>TCA
    Regional News</i>.
    <a href="https://www.pressherald.com/2016/07/03/as-popular-acadia-turns-100-theres-no-room-at-the-top/">https://www.pressherald.com/2016/07/03/as-popular-acadia-turns-100-theres-no-room-at-the-top/</a>

Ilesanmi, A. E., &amp; Ilesanmi, T. O. (2021). Methods for image denoising using
    convolutional neural network: A review. <i>Complex &amp; Intelligent Systems</i>, <i>7</i>(5),
    2179–2198. <a href="https://doi.org/10.1007/s40747-021-00428-4">https://doi.org/10.1007/s40747-021-00428-4</a>

Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., &amp; Kalfaoglu, M. E. (2026).
    <i>Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models</i> (Version 1). arXiv.
    <a href="https://doi.org/10.48550/ARXIV.2606.03748">https://doi.org/10.48550/ARXIV.2606.03748</a>

Jones, T. E., Yang, Y., &amp; Yamamoto, K. (2018). Comparing automated and manual visitor
    monitoring methods: Integrating parallel datasets on Mount Fuji’s North Face.
    <i>Journal of Park and Recreation Administration</i>, <i>36</i>(1), 22–38.
    <a href="https://doi.org/10.18666/JPRA-2018-V36-I1-7976">https://doi.org/10.18666/JPRA-2018-V36-I1-7976</a>

Li, N. (2023). Ethical considerations in artificial intelligence: A comprehensive
    disccusion from the perspective of computer vision. <i>SHS Web of Conferences</i>, <i>179</i>,
    04024. <a href="https://doi.org/10.1051/shsconf/202317904024">https://doi.org/10.1051/shsconf/202317904024</a>

Liu, Z., Dai, C., &amp; Li, X. (2024). An electric bicycle tracking algorithm for improved
    traffic management. <i>Heliyon</i>, <i>10</i>(13), e32708.
    <a href="https://doi.org/10.1016/j.heliyon.2024.e32708">https://doi.org/10.1016/j.heliyon.2024.e32708</a>

Louridas, P., &amp; Ebert, C. (2016). Machine learning. <i>IEEE Software</i>, <i>33</i>(5), 110–115.
    <a href="https://doi.org/10.1109/MS.2016.114">https://doi.org/10.1109/MS.2016.114</a>

Lu, R., Zhang, H., &amp; Zhang, Z. (2025). Revealing urban functionality through vehicle
    dwell patterns: A data-driven approach. <i>2025 IEEE 28th International Conference on
    Intelligent Transportation Systems (ITSC)</i>, 2544–2549.
    <a href="https://doi.org/10.1109/ITSC60802.2025.11423278">https://doi.org/10.1109/ITSC60802.2025.11423278</a>

Lune, H., &amp; Berg, B. L. (2017). <i>Qualitative research methods for the social sciences</i>
    (Ninth edition, global edition). Pearson.

Lupp, G., Kantelberg, V., Förster, B., Honert, C., Naumann, J., Markmann, T., &amp; Pauleit,
    S. (2021). Visitor counting and monitoring in forests using camera traps: A case
    study from Bavaria (Southern Germany). <i>Land (Basel)</i>, <i>10</i>(7), 736.
    <a href="https://doi.org/10.3390/land10070736">https://doi.org/10.3390/land10070736</a>

<i>Managing congestion: A toolkit for parks</i>. (2020, December). National Park Service.
    <a href="https://www.nps.gov/orgs/1548/upload/Congestion_Management_2021-508.pdf">https://www.nps.gov/orgs/1548/upload/Congestion_Management_2021-508.pdf</a>

Manning, R., Jacobi, C., &amp; Marion, J. L. (2006). Recreation monitoring at Acadia
    National Park. <i>The George Wright Forum</i>, <i>23</i>(2), 59–72. JSTOR.

Miguel, J., Mendonça, P., Quelhas, A., Caldeira, J. M. L. P., &amp; Soares, V. N. G. J.
    (2024). The development of a prototype solution for collecting information on
    cycling and hiking trail users. <i>Information</i>, <i>15</i>(7), 389.
    <a href="https://doi.org/10.3390/info15070389">https://doi.org/10.3390/info15070389</a>

Mohammed, S., Budach, L., Feuerpfeil, M., Ihde, N., Nathansen, A., Noack, N., Patzlaff,
    H., Naumann, F., &amp; Harmouch, H. (2025). The effects of data quality on machine
    learning performance on tabular data. <i>Information Systems</i>, <i>132</i>, 102549.
    <a href="https://doi.org/10.1016/j.is.2025.102549">https://doi.org/10.1016/j.is.2025.102549</a>

Monika, Singh, P., &amp; Chand, S. (2023). Computer vision-based framework for pedestrian
    movement direction recognition. <i>Journal of Intelligent &amp; Fuzzy Systems</i>, <i>44</i>(5),
    8015–8027. <a href="https://doi.org/10.3233/JIFS-223283">https://doi.org/10.3233/JIFS-223283</a>

<i>Monitoring guidebook</i>. (2019, June). Interagency Visitor Use Management Council.
    <a href="https://visitorusemanagement.nps.gov/Content/documents/508_final_Monitoring_Guidebook_Edition_One_IVUMC.pdf">https://visitorusemanagement.nps.gov/Content/documents/508_final_Monitoring_Guidebook_Edition_One_IVUMC.pdf</a>

Nadadur, D., Haralick, R. M., &amp; Gustafson, D. E. (2005). A Bayesian framework for noise
    covariance estimation using the facet model. <i>IEEE Transactions on Image Processing</i>,
    <i>14</i>(11), 1902–1917. <a href="https://doi.org/10.1109/TIP.2005.854480">https://doi.org/10.1109/TIP.2005.854480</a>

National Park Service. (2025). <i>Annual park ranking report for recreation visits in:
    2025</i>. Integrated Resource Management Applications.
    <a href="https://irma.nps.gov/Stats/SSRSReports/National%20Reports/Annual%20Park%20Ranking%20Report%20(1979%20-%20Last%20Calendar%20Year)">https://irma.nps.gov/Stats/SSRSReports/National%20Reports/Annual%20Park%20Ranking%20Report%20(1979%20-%20Last%20Calendar%20Year)</a>

<i>Quarterly acreage reports</i>. (2026, March 31). National Park Service.
    <a href="https://www.nps.gov/subjects/lwcf/acreagereports.htm">https://www.nps.gov/subjects/lwcf/acreagereports.htm</a>

Razzaque, T., Hussain, H., Ahmmad, R., &amp; Siddique, S. (2024). Enhancing Safety and
    Collision Avoidance in Autonomous Vehicles through Pose Estimation Techniques. <i>2024
    13th International Conference on Electrical and Computer Engineering (ICECE)</i>,
    385–390. <a href="https://doi.org/10.1109/ICECE64886.2024.11024749">https://doi.org/10.1109/ICECE64886.2024.11024749</a>

Singh, S., &amp; Singh, K. (2025). Pattern recognition and image segmentation based on some
    novel fuzzy similarity measures. <i>Journal of Experimental &amp; Theoretical Artificial
    Intelligence</i>, <i>37</i>(8), 1453–1480. <a href="https://doi.org/10.1080/0952813X.2024.2440662">https://doi.org/10.1080/0952813X.2024.2440662</a>

Tian, B., Tang, M., &amp; Wang, F.-Y. (2015). Vehicle detection grammars with partial
    occlusion handling for traffic surveillance. <i>Transportation Research Part C:
    Emerging Technologies</i>, <i>56</i>, 80–93. <a href="https://doi.org/10.1016/j.trc.2015.02.020">https://doi.org/10.1016/j.trc.2015.02.020</a>

Viñals, M. J., Orozco Carpio, P. R., Teruel, P., &amp; Gandía-Romero, J. M. (2024).
    Real-time monitoring of visitor carrying capacity in crowded historic streets
    through digital technologies. <i>Urban Science</i>, <i>8</i>(4), 190.
    <a href="https://doi.org/10.3390/urbansci8040190">https://doi.org/10.3390/urbansci8040190</a>

Waelen, R. A. (2023). The ethics of computer vision: An overview in terms of power. <i>AI
    and Ethics</i>, <i>4</i>(2), 353–362. <a href="https://doi.org/10.1007/s43681-023-00272-x">https://doi.org/10.1007/s43681-023-00272-x</a>

Wang, Q., Liu, T., &amp; Li, R. (2025). <i>Artificial intelligence and environmental
    sustainability: Investigating the AI‐EKC Nexus for SDG 7 and SDG 13</i>.
    <a href="https://doi.org/10.1002/sd.70294">https://doi.org/10.1002/sd.70294</a>

Watson, A. E., Cole, D. N., Turner, D. L., &amp; Reynolds, P. S. (2000). <i>Wilderness
    recreation use estimation: A handbook of methods and systems</i>. United States
    Department of Agriculture Forest Service.
    <a href="https://www.fs.usda.gov/rm/pubs/rmrs_gtr056.pdf">https://www.fs.usda.gov/rm/pubs/rmrs_gtr056.pdf</a>

Xue, W., Sun, J., Liang, F., Hou, J., Yang, Y., Shang, W., Chen, X., Gradoni, G., &amp;
    Huang, Y. (2025). Jensen–Shannon divergence hypothesis test for determining
    reverberation chamber field distribution. <i>IEEE Transactions on Antennas and
    Propagation</i>, <i>73</i>(9), 6855–6870. <a href="https://doi.org/10.1109/TAP.2025.3574921">https://doi.org/10.1109/TAP.2025.3574921</a>

Yohannes, E., Lin, C.-Y., Shih, T. K., Thaipisutikul, T., Enkhbat, A., &amp; Utaminingrum,
    F. (2023). An improved speed estimation using deep homography transformation
    regression network on monocular videos. <i>IEEE Access</i>, <i>11</i>, 5955–5965.
    <a href="https://doi.org/10.1109/ACCESS.2023.3236512">https://doi.org/10.1109/ACCESS.2023.3236512</a>

Zhao, G., Takafumi, M., Shoji, K., &amp; Kenji, M. (2012). Video based estimation of
    pedestrian walking direction for pedestrian protection system. <i>Journal of
    Electronics (China)</i>, <i>29</i>(1–2), 72–81. <a href="https://doi.org/10.1007/s11767-012-0814-y">https://doi.org/10.1007/s11767-012-0814-y</a>

Zualkernan, I., Dhou, S., Judas, J., Sajun, A. R., Gomez, B. R., &amp; Hussain, L. A.
    (2022). An IoT system using deep learning to classify camera trap images on the
    edge. <i>Computers</i>, <i>11</i>(1), 13. <a href="https://doi.org/10.3390/computers11010013">https://doi.org/10.3390/computers11010013</a>
</pre>

</details>
