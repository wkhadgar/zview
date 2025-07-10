"""
 :file: mcu_scraper.py
 :author: Paulo Santos (pauloroberto.santos@edge.ufal.br)
 :brief: Utilitário de coleta de dados na MCU.
 :version: 0.1
 :date: 13-06-2025

 :copyright: Copyright (c) Paulo Santos 2025
"""

import pylink


class MCUScraper:
    def __init__(self, target_mcu: str, *, ram_start=0x2000_0000, ram_size_k=32):
        # Inicia o JLink.
        self.jlink = pylink.JLink()

        self.target_mcu = target_mcu
        self.ram_start = ram_start
        self.ram_end = ram_start + (ram_size_k * 1024)

        self.target_marker_addr = None

    def __enter__(self):
        self.connect()

        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        try:
            self.jlink.open()
        except pylink.JLinkException:
            raise Exception("\nNão foi possível iniciar o JLink, ele está conectado?")

        # Conecta com a MCU.
        self.jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
        print(f"Pesquisando MCU via {self.jlink.product_name}")

        try:
            self.jlink.connect(self.target_mcu)
        except pylink.JLinkException:
            self.jlink.close()
            raise Exception(f"\nNão foi possível conectar com a MCU [{self.target_mcu}], verifique suas conexões.")

    def disconnect(self):
        self.jlink.close()

    def find_marker(self, marker: str) -> int | None:
        dot_progress_step = 0
        ram_pos = self.ram_start
        dot_step = (self.ram_end - ram_pos) // 50
        while ram_pos < self.ram_end:
            data, = self.jlink.memory_read8(ram_pos, 1)

            if ram_pos % dot_step == 0:
                dot_progress_step += 1
                print(("▰" * dot_progress_step).ljust(50, "▱"), f"Buscando {marker}", end="\r")

            if chr(data) == marker[0]:
                marker_confirm = self.jlink.memory_read8(ram_pos, len(marker))
                marker_confirm = ''.join([chr(v) for v in marker_confirm])
                if marker_confirm == marker:
                    ram_pos += len(marker)
                    print("▰" * 50)
                    print(f"\n{marker} localizado em: 0x{ram_pos:X}")
                    break

            ram_pos += 1

        if ram_pos == self.ram_end:
            self.jlink.close()
            raise Exception(f"\nMarcador não encontrado.\n"
                            f"O formato do marcador deve ser '{marker}' e estar contido na RAM da MCU\n")

        self.target_marker_addr = ram_pos
        return ram_pos
