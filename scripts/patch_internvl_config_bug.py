"""Patches a known bug in InternVL's own configuration_internvl_chat.py that
breaks under transformers>=4.44 (their own code comment acknowledges this:
"There might still be a bug in transformers version 4.44 and above").
transformers' PretrainedConfig.to_diff_dict() instantiates a blank
self.__class__() for repr/diffing purposes, which hits InternVLChatConfig's
own None -> {'architectures': ['']} placeholder and raises unconditionally.

Run this once after cloning https://github.com/OpenGVLab/InternVL.git,
before using anything that touches InternVLChatConfig (training or the
LoRA-reload path in extract_lora_adapter.py) against a modern transformers
version.

Usage: python3 scripts/patch_internvl_config_bug.py [path/to/InternVL]
(defaults to ./InternVL if not given)
"""

import sys

OLD = (
    "        else:\n"
    "            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))"
)
NEW = (
    "        elif not llm_config['architectures'] or not llm_config['architectures'][0]:\n"
    "            # Patched (see scripts/patch_internvl_config_bug.py): transformers >=4.44's\n"
    "            # PretrainedConfig.to_diff_dict() instantiates a blank self.__class__() for\n"
    "            # repr/diffing purposes, which hits this class's own None->{'architectures': ['']}\n"
    "            # placeholder and used to raise here unconditionally (acknowledged as a known\n"
    "            # issue in this file's own TODO above). Falling back to Qwen2Config for this\n"
    "            # diagnostic-only blank instance is safe: real training/inference always\n"
    "            # constructs this with a real llm_config loaded from an actual checkpoint,\n"
    "            # never hits this branch with real data.\n"
    "            self.llm_config = Qwen2Config()\n"
    "        else:\n"
    "            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))"
)


def main() -> None:
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "InternVL"
    path = f"{repo_root}/internvl_chat/internvl/model/internvl_chat/configuration_internvl_chat.py"
    with open(path) as f:
        content = f.read()
    if NEW in content:
        print(f"{path} already patched, nothing to do")
        return
    if OLD not in content:
        raise SystemExit(f"expected snippet not found in {path} -- InternVL's code may have changed, check manually")
    with open(path, "w") as f:
        f.write(content.replace(OLD, NEW))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
