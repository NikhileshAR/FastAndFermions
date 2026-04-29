Repository Structure:-
```
FastAndFermions/
├── consolidated_data.csv       
│
├── phase1/full final/
│   └── phase1.py               # Analyzes Gaussian core width σ vs √t (Fig 1)
│
├── phase2/                     # Distribution shape & tail analysis
│   ├── phase2_v2.py            # Log-scale distributions vs Highland predictions (Fig 2)
│   ├── blok2/final/blok2.py    # Tail ratio
│   ├── blok3/blok3.py          # σ_core / σ_Highland ratio vs thickness
│   └── blok4/blok4.py          # Heatmaps and fixed-angle tail ratios
│
├── phase3/
│   └── phase3.py               # Power-law extraction & log-log fits
│
└── Simulation-GEANT4/          # Submodule (BeamlineProject) containing C++ source
```

Getting Started:-
1. Python Analysis
```
  pip install numpy matplotlib scipy

  cd "phase1/full final"
  python phase1.py

  cd phase2
  python phase2_v2.py       
  python blok2/final/blok2.py
  python blok3/blok3.py
  python blok4/blok4.py 

  cd phase3
  python phase3.py
```
phase1.py reads `consolidated_data.csv` from the repo root.  
All scripts save output figures as `.png` in their local directory.

2. GEANT4 Simulation
See the `Simulation-GEANT4` submodule (BeamlineProject) for the full C++ source.
```
  cd ~/BeamlineProject 
  
  rm -rf build 
  mkdir build 
  cd build 
  cmake .. 
  make -j4
  
  ./scattering
  root output.root
  ScatteringAngle_Full->Draw(); gPad->SetLogy();
  ScatteringAngle->Draw(); gPad->SetLogy();
  TTree* t = (TTree*)_file0->Get("scattering"); t->Draw("theta"); t->Draw("energy"); t->Draw("theta","","");
```

