from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'pallet_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['image_publisher = pallet_detection.publish_camera_feed:main',
                            'image_subscriber = pallet_detection.process_camera_feed:main',
                            'model_ready_waiter = pallet_detection.wait_for_model:main',
        ],
    },
)
