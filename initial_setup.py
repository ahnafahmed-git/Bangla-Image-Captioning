import pip
#Installing bnlp_toolkit
pip install -U bnlp_toolkit

#Setting up Bengali Fonts
!apt-get update -qq
!apt-get install -y fonts-noto fonts-noto-cjk
!fc-cache -fv
print("Fonts installed")
