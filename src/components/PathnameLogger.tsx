"use client"

import { usePathname } from 'next/navigation'
import React from 'react'

export function PathnameLogger() {
  const pathname = usePathname();

  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      console.log(`Navegação: ${pathname}`);
    }
  }, [pathname]);

  return null; // Este componente não renderiza nada visível
}