#!/bin/bash
mkdir external_validation
cd external_validation

wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv

for i in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10'

    do
        wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/vol$i.7z
        mkdir vol$i; mv vol$i.7z vol$i
        cd vol$i
        7za e vol$i.7z; rm vol$i.7z
        cd ..
    done

cd ..
wget -O mrnet_data  "https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FMRNet-v1.0.zip&h=1092de8238b6deef177f6b987fe38d716b1416b597272b95d021b780e2d3fd86&v=1&xid=25262b0e0d&uid=55365305&pool=contact_facing&subject=MRNet-v1.0%3A+Subscription+Confirmed"

unzip mrnet_data
rm mrnet_data
mv MRNet-v1.0/ mrnet_data

