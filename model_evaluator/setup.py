from setuptools import find_packages, setup

package_name = 'model_evaluator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', 'launch/launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Julius Schulte',
    maintainer_email='j.schulte-1@sms.ed.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'model_evaluator = model_evaluator.model_evaluator:main'
        ],
    },
)
