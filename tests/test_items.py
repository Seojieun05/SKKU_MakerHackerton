# SPDX-License-Identifier: AGPL-3.0-only
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from seat_monitor.camera import FrameSource
from seat_monitor.core import Detection, SeatState
from seat_monitor.items_run import SceneDetector, add_item_fields, items_parser, run
from seat_monitor.status import snapshot


def tensor(value):
    obj = MagicMock()
    obj.cpu.return_value.tolist.return_value = value
    return obj


class ItemsTests(unittest.TestCase):
    def test_one_inference_uses_names_and_separate_thresholds(self):
        detector = object.__new__(SceneDetector)
        detector.confidence, detector.imgsz = .4, 416
        detector.model = MagicMock()
        detector.model.names = {9:'person',42:'laptop',3:'chair',17:'cup',19:'book'}
        boxes = types.SimpleNamespace(
            xyxyn=tensor([[0,0,1,1]]*7),
            conf=tensor([.6,.35,.9,.2,.39,.8,.7]),
            cls=tensor([9,42,3,42,9,17,19]))
        detector.model.predict.return_value = [types.SimpleNamespace(boxes=boxes)]
        people, items = detector.detect_scene(None,.3)
        self.assertEqual([d.confidence for d in people],[.6])
        self.assertEqual([d.confidence for d in items],[.35,.8,.7])
        detector.model.predict.assert_called_once()
        kwargs = detector.model.predict.call_args.kwargs
        self.assertEqual(set(kwargs['classes']),{9,42,17,19})
        self.assertFalse(kwargs['save'])

    def test_empty_detections(self):
        detector = object.__new__(SceneDetector)
        detector.confidence, detector.imgsz = .4, 416
        detector.model = MagicMock()
        detector.model.names = ['person','laptop']
        detector.model.predict.return_value = [types.SimpleNamespace(boxes=None)]
        self.assertEqual(detector.detect_scene(None,.3),([],[]))

    def test_missing_item_classes_fail_explicitly(self):
        detector = object.__new__(SceneDetector)
        detector.model = types.SimpleNamespace(names={0:'person'})
        with self.assertRaisesRegex(ValueError,'personal-item'):
            detector.detect_scene(None,.3)

    def test_presence_preserves_original_person_semantics(self):
        for person_state, item_state, expected in (
            ('OCCUPIED','EMPTY','PERSON_PRESENT'),
            ('EMPTY','OCCUPIED','ITEMS_ONLY'),
            ('EMPTY','EMPTY','EMPTY'),
            ('UNKNOWN','OCCUPIED','UNKNOWN'),
            ('EMPTY','UNKNOWN','UNKNOWN'),
        ):
            with self.subTest(expected=expected,person=person_state,item=item_state):
                human, item = SeatState(), SeatState()
                human.state, item.state = person_state,item_state
                data = snapshot({'A01':human},{'A01':0},0)
                add_item_fields(data,{'A01':item},{'A01':.8},'ok')
                self.assertEqual(data['meaning'],'person_presence_only')
                self.assertEqual(data['seats'][0]['state'],person_state)
                self.assertEqual(data['seats'][0]['presence_state'],expected)

    def test_unhealthy_results_never_report_known_items(self):
        human,item=SeatState(),SeatState()
        human.state=item.state='OCCUPIED'
        for health in ('error','stopped','inference_too_slow'):
            data=snapshot({'A01':human},{'A01':.8},0,health)
            add_item_fields(data,{'A01':item},{'A01':.8},health)
            self.assertIsNone(data['seats'][0]['has_item'])
            self.assertEqual(data['seats'][0]['presence_state'],'UNKNOWN')

    def test_explicit_sensor_mode_and_unsupported_mode_cleanup(self):
        camera=MagicMock()
        camera.sensor_modes=[{'size':(2304,1296),'bit_depth':10}]
        module=types.SimpleNamespace(Picamera2=MagicMock(return_value=camera))
        with patch.dict('sys.modules',{'picamera2':module}), patch('seat_monitor.camera.time.sleep'):
            with FrameSource('picamera2',960,720,sensor_size=(2304,1296)):
                pass
            self.assertEqual(camera.create_video_configuration.call_args.kwargs['sensor'],
                             {'output_size':(2304,1296),'bit_depth':10})
            camera.reset_mock()
            with self.assertRaisesRegex(ValueError,'sensor mode'):
                with FrameSource('picamera2',960,720,sensor_size=(1,1)):
                    pass
            camera.close.assert_called_once()

    def setup_runtime(self,directory):
        root=Path(directory)
        payload={'schema_version':1,'frame_size':[960,720],
                 'seats':[{'id':'A01','polygon':[[0,0],[1,0],[1,1],[0,1]]}]}
        for name in ('people.json','items.json'):
            (root/name).write_text(json.dumps(payload))
        return items_parser().parse_args(['run','--config',str(root/'people.json'),
            '--items-config',str(root/'items.json'),'--status-file',str(root/'status.json'),
            '--width','960','--height','720','--empty-seconds','1'])

    def test_runtime_items_only_then_empty_and_shutdown_unknown(self):
        frame=types.SimpleNamespace(shape=(720,960,3))
        camera=MagicMock()
        camera.__enter__.return_value=camera
        camera.is_video=True
        camera.read.side_effect=[frame]*5+[None]
        observations=iter([0,1,2,3,4])
        type(camera).observation_time=property(lambda _: next(observations))
        detector=MagicMock()
        item=Detection((.1,.1,.4,.4),.9)
        detector.detect_scene.side_effect=[([],[item])]*3+[([],[])]*2
        with tempfile.TemporaryDirectory() as directory:
            args=self.setup_runtime(directory)
            stream=io.StringIO()
            with patch('seat_monitor.items_run.FrameSource',return_value=camera), \
                 patch('seat_monitor.items_run.SceneDetector',return_value=detector), \
                 contextlib.redirect_stdout(stream):
                self.assertEqual(run(args),0)
            messages=[json.loads(line) for line in stream.getvalue().splitlines()]
            presences=[m['seats'][0]['presence_state'] for m in messages]
            self.assertIn('ITEMS_ONLY',presences)
            self.assertIn('EMPTY',presences)
            final=json.loads(Path(args.status_file).read_text())
            self.assertEqual(final['health'],'end_of_video')
            self.assertIsNone(final['seats'][0]['has_item'])
            self.assertIsNone(final['seats'][0]['has_person'])

    def test_slow_inference_invalidates_both_kinds(self):
        camera=MagicMock()
        camera.__enter__.return_value=camera
        camera.is_video=True
        camera.read.return_value=types.SimpleNamespace(shape=(720,960,3))
        camera.observation_time=0
        detector=MagicMock()
        detector.detect_scene.return_value=([Detection((0,0,1,1),.9)],[Detection((0,0,1,1),.9)])
        with tempfile.TemporaryDirectory() as directory:
            args=self.setup_runtime(directory)
            args.max_frames=1
            stream=io.StringIO()
            with patch('seat_monitor.items_run.FrameSource',return_value=camera), \
                 patch('seat_monitor.items_run.SceneDetector',return_value=detector), \
                 patch('seat_monitor.items_run.time.monotonic',side_effect=[0,0,10]), \
                 contextlib.redirect_stdout(stream):
                run(args)
            messages=[json.loads(line) for line in stream.getvalue().splitlines()]
            slow=next(m for m in messages if m['health']=='inference_too_slow')
            self.assertIsNone(slow['seats'][0]['has_person'])
            self.assertIsNone(slow['seats'][0]['has_item'])

    def test_config_mismatch_and_invalid_sensor_options(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.setup_runtime(directory)
            args.sensor_width=2304
            with self.assertRaisesRegex(ValueError,'both'):
                run(args)
            args.sensor_width=None
            payload=json.loads(args.items_config.read_text())
            payload['seats'][0]['id']='A02'
            args.items_config.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'IDs'):
                run(args)


if __name__=='__main__':
    unittest.main()
