import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pymss import MSSeparator, get_separation_logger

def test1():
    logger = get_separation_logger()
    logger.info("test1")

    separator = MSSeparator(
        model_type='htdemucs', 
        model_path='pretrain/multi_stem_models/HTDemucs4_6stems.th',
        config_path='configs/multi_stem_models/config_htdemucs_6stems.yaml',
        device='cpu',
        device_ids=[0],
        output_format='wav',
        store_dirs={
            "vocals": "./output/vocals",
            "other": None,
        },
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()

def test2():
    logger = get_separation_logger()
    logger.info("test2")

    separator = MSSeparator(
        model_type="bandit",
        model_path='pretrain/multi_stem_models/model_bandit_plus_dnr_sdr_11.47.chpt',
        config_path="configs/multi_stem_models/config_dnr_bandit_bsrnn_multi_mus64.yaml",
        device='cpu',
        device_ids=[0],
        output_format='wav',
        store_dirs={
            "speech": "./output/speech",
            "effects": None,
            "music": "./output/music",
        },
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()

def test3():
    logger = get_separation_logger()
    logger.info("test3")

    separator = MSSeparator(
        model_type="mdx23c",
        model_path='pretrain/multi_stem_models/model_mdx23c_ep_168_sdr_7.0207.ckpt',
        config_path="configs/multi_stem_models/config_musdb18_mdx23c.yaml",
        device='mps',
        device_ids=[0],
        output_format='wav',
        store_dirs={
            "vocals": "./output/vocals",
            "other": "./output/other",
        },
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()

def test4():
    logger = get_separation_logger()
    logger.info("test4")

    separator = MSSeparator(
        model_type="bs_roformer",
        model_path='pretrain/single_stem_models/deverb_bs_roformer_8_256dim_8depth.ckpt',
        config_path="configs/single_stem_models/deverb_bs_roformer_8_256dim_8depth.yaml",
        device='mps',
        device_ids=[0],
        output_format='wav',
        store_dirs="output",
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()

def test5():
    logger = get_separation_logger()
    logger.info("test5")

    separator = MSSeparator(
        model_type="apollo",
        model_path='pretrain/single_stem_models/Apollo_LQ_MP3_restoration.ckpt',
        config_path="configs/single_stem_models/config_apollo.yaml",
        device='mps',
        device_ids=[0],
        output_format='wav',
        store_dirs="output",
        logger=logger,
    )

    separator.process_folder("./mp3")
    separator.del_cache()    

def test6():
    logger = get_separation_logger()
    logger.info("test6")

    separator = MSSeparator(
        model_type="bs_roformer",
        model_path='pretrain/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        config_path="configs/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        device='mps',
        device_ids=[0],
        output_format='wav',
        store_dirs="output",
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache() 

def test7():
    logger = get_separation_logger()
    logger.info("test7")

    separator = MSSeparator(
        model_type="scnet",
        model_path='pretrain/multi_stem_models/scnet_checkpoint_musdb18.ckpt',
        config_path="configs/multi_stem_models/config_musdb18_scnet.yaml",
        device='mps',
        device_ids=[0],
        output_format='wav',
        store_dirs="output",
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()  

def test8():
    logger = get_separation_logger()
    logger.info("test8")

    separator = MSSeparator(
        model_type="mel_band_roformer",
        model_path='pretrain/vocal_models/mel_band_roformer_vocals_becruily.ckpt',
        config_path="configs/vocal_models/config_vocals_becruily.yaml",
        device='cpu',
        device_ids=[0],
        output_format='wav',
        store_dirs={
            "vocals": "./output/vocals",
        },
        logger=logger,
    )

    separator.process_folder("./input")
    separator.del_cache()            

if __name__ == "__main__":
    test8()
